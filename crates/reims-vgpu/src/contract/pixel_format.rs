//! Metal pixel-format helpers (port of `host/utils/reims-vgpu-pixel-format`).

use crate::contract::endian::{ld16, st16};

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
pub const R32F_BPP: u32 = 4;
pub const RG16F_BPP: u32 = 4;
pub const RGBA16_BPP: u32 = 8;
pub const RGBA16F_BPP: u32 = RGBA16_BPP;
pub const RGBA32_BPP: u32 = 16;
pub const RGBA32F_BPP: u32 = RGBA32_BPP;
pub const R32_BPP: u32 = 4;

// MTLPixelFormat values (Metal.framework Headers/MTLPixelFormat.h).
pub const MTL_FORMAT_A8_UNORM: u16 = 0x01;
pub const MTL_FORMAT_R8_UNORM: u16 = 0x0a;
/// `MTLPixelFormatR16Unorm = 20`. Two bytes, one channel.
///
/// Sits five below `R16Float` in the same enum, which this table already
/// carries at `0x19` — so the value is read off the header, not inferred from a
/// measurement. A live macOS desktop binds it as a 3840x2160 type-5 view, and
/// without it here that view refused 100 times in one boot.
pub const MTL_FORMAT_R16_UNORM: u16 = 0x14;
pub const MTL_FORMAT_R16_FLOAT: u16 = 0x19;
pub const MTL_FORMAT_RG8_UNORM: u16 = 0x1e;
/// `MTLPixelFormatRG16Unorm = 60`. Four bytes, two channels of 16-bit unorm.
///
/// The chroma half of a 10-bit biplanar 4:2:0 surface, whose luma half is
/// [`MTL_FORMAT_R16_UNORM`]. Measured together on one live macOS desktop: a
/// `x420` surface (FourCC `0x78343230`, `planes=2`) publishes plane 0 as
/// 3840x2160 at exactly two bytes per texel and plane 1 as 1920x1080 — half in
/// both axes, which is what 4:2:0 subsampling means — and a full-screen quad
/// samples the two at fragment texture indices 3 and 5, doing the YCbCr-to-RGB
/// conversion itself.
///
/// Value read off `MTLPixelFormat.h` like the rest of this table: `RG16Float` is
/// already here at `0x41` (65), and `RG16Unorm` is 60.
pub const MTL_FORMAT_RG16_UNORM: u16 = 0x3c;
/// `MTLPixelFormatRG16Uint = 63`. Four bytes, two channels of 16-bit unsigned
/// integer.
///
/// Identified from the guest shader's decoded argument type and translated SPIR-V
/// image type, both of which treat the image as unsigned integer texels.
///
/// A distance field: the shader fetches the texel, treats the two channels as a
/// texel coordinate, and never combines the image with a sampler. So it needs no
/// filtering, which matters because Vulkan does not guarantee linear filtering
/// for integer formats.
///
/// Deliberately absent from [`render_target_class`], [`texel_to_rgba8`] and
/// [`rgba8_to_texel`]. Those rails carry colour through 8-bit unorm LUTs, and a
/// coordinate stored in 16 unsigned bits does not survive that — it would come
/// back quantized and reinterpreted as a normalized fraction. Declining a
/// colour-target bind by name is the honest answer until the render rail can
/// carry integer texels; producing wrong pixels would be worse than producing
/// none.
pub const MTL_FORMAT_RG16_UINT: u16 = 0x3f;
pub const MTL_FORMAT_R32_UINT: u16 = 0x35;
pub const MTL_FORMAT_R32_SINT: u16 = 0x36;
pub const MTL_FORMAT_R32_FLOAT: u16 = 0x37;
pub const MTL_FORMAT_RG16_FLOAT: u16 = 0x41;
pub const MTL_FORMAT_RGBA8_UNORM: u16 = 0x46;
pub const MTL_FORMAT_RGBA8_UNORM_SRGB: u16 = 0x47;
pub const MTL_FORMAT_RGBA8_UINT: u16 = 0x49;
pub const MTL_FORMAT_RGBA8_SINT: u16 = 0x4a;
pub const MTL_FORMAT_BGRA8_UNORM: u16 = 0x50;
pub const MTL_FORMAT_BGRA8_UNORM_SRGB: u16 = 0x51;
/// Packed RGB9E5 shared-exponent float. 32-bit texels.
pub const MTL_FORMAT_RGB9E5_FLOAT: u16 = 0x5d;
/// Packed 10-bit-per-channel BGR with 2-bit alpha. 32-bit texels.
///
/// The wide-gamut format Tahoe's compositor draws its window content in, and
/// the single largest source of declined views on the x86 rail before it was
/// modelled: 424 `type5_view_format_bpp` refusals in one boot, all of it text
/// and window chrome that then rendered wrong.
///
/// Four bytes per texel is measured from the guest's own allocation rather than
/// assumed from the name. The guest describes these surfaces as
/// `fmt=0x6c313072` — FourCC `'l10r'`, CoreVideo's 30-bit little-endian packed
/// wide-gamut RGB — and its rows are `bpr=1472` for `w=365` and `bpr=1536` for
/// `w=370`, i.e. `w * 4` rounded up to 64 bytes in both cases.
pub const MTL_FORMAT_BGR10A2_UNORM: u16 = 0x5e;
/// Four 16-bit unorm channels. 64-bit texels.
///
/// Sits directly below the `RGBA16Uint`/`RGBA16Float` pair already modelled
/// here, and the guest's own allocation confirms the width: a `384x384` surface
/// arrives with `bpr=3072` and `alloc=1179648`, i.e. exactly `384 * 8` per row
/// and `384 * 384 * 8` in total. Declining it cost 93 sampled loads per boot
/// (`linear_load_view_bpp_mismatch`), each one a texture that then sampled as
/// nothing.
pub const MTL_FORMAT_RGBA16_UNORM: u16 = 0x6e;
pub const MTL_FORMAT_RGBA16_UINT: u16 = 0x71;
pub const MTL_FORMAT_RGBA16_FLOAT: u16 = 0x73;
pub const MTL_FORMAT_RGBA32_UINT: u16 = 0x7b;
pub const MTL_FORMAT_RGBA32_FLOAT: u16 = 0x7d;
// Depth / stencil (Metal.framework Headers/MTLPixelFormat.h).
pub const MTL_FORMAT_DEPTH16_UNORM: u16 = 250;
pub const MTL_FORMAT_DEPTH32_FLOAT: u16 = 252;
pub const MTL_FORMAT_STENCIL8: u16 = 253;
pub const MTL_FORMAT_DEPTH24_UNORM_STENCIL8: u16 = 255;
pub const MTL_FORMAT_DEPTH32_FLOAT_STENCIL8: u16 = 260;
pub const MTL_FORMAT_X32_STENCIL8: u16 = 261;
pub const MTL_FORMAT_X24_STENCIL8: u16 = 262;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageImageSelector {
    Rgba8Uint = 0,
    Rgba8Sint = 1,
    Rgba16Uint = 2,
    Rgba16Float = 3,
    Rgba32Float = 4,
    Rgba8Unorm = 5,
    Bgra8Unorm = 6,
    R16Float = 7,
    Rg16Float = 8,
    R8Unorm = 9,
    Rg8Unorm = 10,
    Rgba32Uint = 11,
    R32Uint = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampledClass {
    A8Unorm,
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Rgba16Float,
}

/// The byte layout of one guest texel on the sampled rails, independent of any
/// host graphics API.
///
/// This is the vocabulary `runtime/` speaks about sampled texels. It is
/// deliberately *narrow*: these are exactly the layouts a CPU-origin upload or
/// an in-place guest gather can hand a sampled image without a conversion pass.
/// A rail that carries the full format set names the host format instead — the
/// engine stores `VkFormat` and can therefore express an sRGB sampled view,
/// which a layout enum by construction cannot.
///
/// It lives in the contract rather than in either the runtime or the backend
/// because both used to hold their own copy of it, with a hand-written mapping
/// between the two that nothing checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLayout {
    /// 4 bytes/texel, guest channel order R,G,B,A — the default CPU-origin
    /// layout, and what every convert-to-RGBA8 loader produces.
    Rgba8,
    /// 4 bytes/texel, guest channel order B,G,R,A. Uploaded as-is so the
    /// sampler swaps channels in hardware and the CPU never runs a per-pixel
    /// swizzle.
    Bgra8,
    /// 1 byte/texel — a biplanar video luma plane, sampled at its native
    /// footprint rather than expanded to RGBA8.
    R8,
    /// 2 bytes/texel — a biplanar video chroma plane, likewise native.
    Rg8,
    /// 2 bytes/texel — a single-channel `float16` texture, sampled natively as
    /// `R16_SFLOAT` (the shader reads `.x`, the other lanes expand to `0,0,1`).
    /// Color-management 1D LUTs (macOS WindowServer's `UberCompositeFragment`
    /// display-profile pass) are stored this way; converting them to unorm8
    /// would quantize the transfer curve, and the CPU `texel_to_rgba8` loader
    /// has no float arm, so this native rail is the only correct path. Not a
    /// four-byte color layout, so it never rides the RGBA8-shaped loaders.
    R16Float,
    /// 2 bytes/texel — a single-channel 16-bit unorm texture, sampled natively
    /// as `R16_UNORM` (the shader reads `.x`, the other lanes expand to
    /// `0,0,1`).
    ///
    /// Same reason as [`Self::R16Float`] for having its own layout rather than
    /// riding the RGBA8 loaders: the CPU `convert_row_to_rgba8` path has no arm
    /// for it, so every bind of such a view was refused. A live macOS desktop
    /// binds one 3840x2160 view of this shape 387 times in a single logged-in
    /// session.
    ///
    /// Not folded into [`Self::Rg8`], which is the same two bytes but a
    /// different reading of them: `R8G8_UNORM` splits the texel into two 8-bit
    /// channels, where this is one 16-bit value.
    R16Unorm,
    /// 4 bytes/texel — two channels of 16-bit unorm, sampled natively as
    /// `R16G16_UNORM`.
    ///
    /// The chroma plane of a 10-bit biplanar 4:2:0 surface. Native for the same
    /// reason as [`Self::Rg8`] is for 8-bit chroma: the shader reads `.rg` and
    /// does its own YCbCr conversion, so expanding to RGBA8 would both cost 2x
    /// the staging bytes and quantize ten bits down to eight.
    Rg16Unorm,
    /// 4 bytes/texel — two channels of 16-bit unsigned integer, sampled natively
    /// as `R16G16_UINT`.
    ///
    /// Integer texels, so this layout exists precisely to keep them off the
    /// RGBA8 rails: the bytes are uploaded verbatim and the shader fetches them
    /// with `OpImageFetch`, which is the only way it is ever accessed. Passing
    /// them through the unorm8 LUTs would quantize a 16-bit coordinate to 8 bits
    /// and then read it as a fraction.
    Rg16Uint,
    /// 4 bytes/texel — a single-channel `float32` texture, sampled natively as
    /// `R32_SFLOAT`. Same color-LUT role as [`Self::R16Float`], but its
    /// linear-filter feature is optional (absent on Apple/MoltenVK), so the
    /// rail that emits this layout must first confirm the host supports it.
    /// Four bytes wide but **not** a colour order, so it stays out of the
    /// RGBA8-shaped loaders and `is_four_byte_color`.
    R32Float,
}

impl TexelLayout {
    /// Bytes occupied by one texel in guest linear storage.
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 => RGBA8_BPP,
            Self::R8 => R8_BPP,
            Self::Rg8 => RG8_BPP,
            Self::R16Float | Self::R16Unorm => R16F_BPP,
            Self::Rg16Unorm | Self::Rg16Uint => RG16F_BPP,
            Self::R32Float => R32F_BPP,
        }
    }

    /// Whether this layout is one of the two four-byte colour orders.
    ///
    /// Several rails admit only these: the RGBA8-shaped diagnostics, the
    /// tight-row loaders and the zero-copy gathers all assume a four-byte
    /// texel. Named once so a rail states which set it takes instead of
    /// re-listing the variants.
    pub fn is_four_byte_color(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    Zero = 0,
    One = 1,
    R = 2,
    G = 3,
    B = 4,
    A = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwizzlePlan {
    pub source: [SwizzleSource; COMPONENT_COUNT],
}

impl Default for SwizzlePlan {
    fn default() -> Self {
        swizzle_identity()
    }
}

const UNORM8_MIN: u8 = 0x00;
const UNORM8_MAX: u8 = 0xff;

const F16_SIGN_MASK: u16 = 0x8000;
const F16_EXP_SHIFT: u32 = 10;
const F16_EXP_MASK: u32 = 0x1f;
const F16_MANT_MASK: u32 = 0x03ff;
const F16_HIDDEN_BIT: u32 = 0x0400;
const F16_EXP_BIAS: i32 = 15;
const F16_EXP_INF_NAN: u32 = F16_EXP_MASK;
const F16_INF_BITS: u16 = 0x7c00;
const F16_SUBNORMAL_EXP_MIN: i32 = -10;
const F16_SUBNORMAL_SHIFT_BASE: i32 = 14;
const F16_F32_SIGN_SHIFT: u32 = 16;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_MASK: u32 = 0xff;
const F32_MANT_MASK: u32 = 0x007f_ffff;
const F32_HIDDEN_BIT: u32 = 0x0080_0000;
const F32_EXP_BIAS: i32 = 127;
const F32_INF_BITS: u32 = 0x7f80_0000;
const F16_F32_MANT_SHIFT: u32 = 13;
const F32_TO_F16_ROUND_BIT: u32 = 0x1000;

pub fn bytes_per_pixel(format: u16) -> Option<u32> {
    Some(match format {
        MTL_FORMAT_A8_UNORM | MTL_FORMAT_R8_UNORM | MTL_FORMAT_STENCIL8 => R8_BPP,
        MTL_FORMAT_R16_UNORM
        | MTL_FORMAT_R16_FLOAT
        | MTL_FORMAT_RG8_UNORM
        | MTL_FORMAT_DEPTH16_UNORM => RG8_BPP,
        MTL_FORMAT_RG16_UNORM | MTL_FORMAT_RG16_UINT | MTL_FORMAT_RG16_FLOAT => RG16F_BPP,
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
        | MTL_FORMAT_BGR10A2_UNORM
        | MTL_FORMAT_DEPTH32_FLOAT
        | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
        | MTL_FORMAT_X24_STENCIL8 => RGBA8_BPP,
        // Depth32Float_Stencil8 / X32_Stencil8: 64-bit cells on Apple Silicon
        // (40-bit logical DS + pad; Metal allocates 8 B/texel for this family).
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 | MTL_FORMAT_X32_STENCIL8 => 8,
        MTL_FORMAT_RGBA16_UNORM | MTL_FORMAT_RGBA16_UINT | MTL_FORMAT_RGBA16_FLOAT => {
            RGBA16_BPP
        }
        MTL_FORMAT_RGBA32_UINT | MTL_FORMAT_RGBA32_FLOAT => RGBA32_BPP,
        _ => return None,
    })
}

/// Whether `format` has a depth plane (for `MTLBlitOptionDepthFromDepthStencil`).
pub fn format_has_depth_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_DEPTH16_UNORM
            | MTL_FORMAT_DEPTH32_FLOAT
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
    )
}

/// Whether `format` has a stencil plane (for `MTLBlitOptionStencilFromDepthStencil`).
pub fn format_has_stencil_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_STENCIL8
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
            | MTL_FORMAT_X32_STENCIL8
            | MTL_FORMAT_X24_STENCIL8
    )
}

/// Linear packing of a combined depth-stencil texel (full cell in guest storage).
///
/// Plane sizes when extracted to a buffer match Metal blit options:
/// depth plane → 4 B (Depth32Float / Depth24 expanded unorm32), stencil → 1 B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencilPacking {
    /// Full texel size in guest linear storage.
    pub full_bpp: u32,
    /// Byte offset of the depth field within the texel (if present).
    pub depth_offset: u32,
    /// Buffer-side depth plane size after Metal extraction.
    pub depth_plane_bpp: u32,
    /// Byte offset of the stencil field within the texel (if present).
    pub stencil_offset: u32,
    pub stencil_plane_bpp: u32,
    /// How depth is stored in the packed texel.
    pub depth_layout: DepthFieldLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthFieldLayout {
    /// Not a depth-bearing format (stencil-only combined / X\*\_Stencil8).
    None,
    /// IEEE-754 binary32 LE at `depth_offset`.
    Float32,
    /// 24-bit unorm in bits \[8:31\] of a LE u32 (stencil in low 8 bits).
    Unorm24High,
}

/// Combined depth-stencil packing for formats that interleave both planes.
///
/// Pure Depth32Float / Stencil8 / Depth16 return `None` (no repack; aspect is identity).
pub fn depth_stencil_packing(format: u16) -> Option<DepthStencilPacking> {
    match format {
        // Apple docs: 40-bit logical (32f + 8); Apple Silicon cells are 8 B.
        // Layout: depth f32 @0, stencil u8 @4, pad @5..7.
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 8,
            depth_offset: 0,
            depth_plane_bpp: 4,
            stencil_offset: 4,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::Float32,
        }),
        // 32-bit cell: stencil in low 8, depth unorm24 in high 24 (Metal/macOS common packing).
        MTL_FORMAT_DEPTH24_UNORM_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 4,
            depth_offset: 0,
            depth_plane_bpp: 4,
            stencil_offset: 0,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::Unorm24High,
        }),
        // X32_Stencil8: same 8 B cell as Depth32Float_Stencil8 without meaningful depth.
        MTL_FORMAT_X32_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 8,
            depth_offset: 0,
            depth_plane_bpp: 0,
            stencil_offset: 4,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::None,
        }),
        // X24_Stencil8: 4 B cell, stencil in low 8.
        MTL_FORMAT_X24_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 4,
            depth_offset: 0,
            depth_plane_bpp: 0,
            stencil_offset: 0,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::None,
        }),
        _ => None,
    }
}

/// Bytes per texel for a blit aspect selection.
///
/// Pure depth/stencil formats: aspect matches full bpp (option is identity).
/// Combined formats: Full uses packed `full_bpp`; depth plane is 4 B; stencil is 1 B.
pub fn blit_aspect_bytes_per_pixel(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
) -> Option<u32> {
    if depth_aspect && stencil_aspect {
        return None;
    }
    if depth_aspect {
        if !format_has_depth_aspect(format) {
            return None;
        }
        if let Some(p) = depth_stencil_packing(format) {
            return if p.depth_plane_bpp != 0 {
                Some(p.depth_plane_bpp)
            } else {
                None
            };
        }
        return Some(match format {
            MTL_FORMAT_DEPTH16_UNORM => 2,
            MTL_FORMAT_DEPTH32_FLOAT => 4,
            _ => return None,
        });
    }
    if stencil_aspect {
        if !format_has_stencil_aspect(format) {
            return None;
        }
        if let Some(p) = depth_stencil_packing(format) {
            return Some(p.stencil_plane_bpp);
        }
        return Some(1);
    }
    // Full texel.
    bytes_per_pixel(format)
}

/// Whether a plane extract/insert pass is required (combined DS + aspect option).
pub fn blit_aspect_needs_repack(format: u16, depth_aspect: bool, stencil_aspect: bool) -> bool {
    if !depth_aspect && !stencil_aspect {
        return false;
    }
    depth_stencil_packing(format).is_some()
}

/// Extract one plane from a packed depth-stencil texel into `dst` (plane-native size).
pub fn extract_depth_stencil_plane(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    texel: &[u8],
    dst: &mut [u8],
) -> bool {
    if depth_aspect == stencil_aspect {
        return false;
    }
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    if texel.len() < p.full_bpp as usize {
        return false;
    }
    if depth_aspect {
        if p.depth_plane_bpp == 0 || dst.len() < p.depth_plane_bpp as usize {
            return false;
        }
        match p.depth_layout {
            DepthFieldLayout::Float32 => {
                let o = p.depth_offset as usize;
                dst[..4].copy_from_slice(&texel[o..o + 4]);
            }
            DepthFieldLayout::Unorm24High => {
                // Packed LE u32: stencil @bits0-7, depth unorm24 @bits8-31.
                // Metal depth buffer plane is 32-bit unorm (depth in low 24).
                let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                let depth24 = packed >> 8;
                dst[..4].copy_from_slice(&depth24.to_le_bytes());
            }
            DepthFieldLayout::None => return false,
        }
        return true;
    }
    // Stencil plane.
    if dst.len() < p.stencil_plane_bpp as usize {
        return false;
    }
    dst[0] = texel[p.stencil_offset as usize];
    true
}

/// Insert one plane into a packed depth-stencil texel (read-modify-write).
///
/// `texel` holds the current full cell (updated in place). `src` is plane-native.
pub fn insert_depth_stencil_plane(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    src: &[u8],
    texel: &mut [u8],
) -> bool {
    if depth_aspect == stencil_aspect {
        return false;
    }
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    if texel.len() < p.full_bpp as usize {
        return false;
    }
    if depth_aspect {
        if p.depth_plane_bpp == 0 || src.len() < p.depth_plane_bpp as usize {
            return false;
        }
        match p.depth_layout {
            DepthFieldLayout::Float32 => {
                let o = p.depth_offset as usize;
                texel[o..o + 4].copy_from_slice(&src[..4]);
            }
            DepthFieldLayout::Unorm24High => {
                let depth24 = u32::from_le_bytes([src[0], src[1], src[2], src[3]]) & 0x00ff_ffff;
                let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                let stencil = packed & 0xff;
                let out = stencil | (depth24 << 8);
                texel[..4].copy_from_slice(&out.to_le_bytes());
            }
            DepthFieldLayout::None => return false,
        }
        return true;
    }
    if src.is_empty() {
        return false;
    }
    texel[p.stencil_offset as usize] = src[0];
    true
}

/// Extract a tight plane row from a strided packed texture row.
pub fn extract_plane_row(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    src_row: &[u8],
    width: u32,
    dst_plane: &mut [u8],
) -> bool {
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    let plane_bpp = if depth_aspect {
        p.depth_plane_bpp
    } else if stencil_aspect {
        p.stencil_plane_bpp
    } else {
        return false;
    } as usize;
    let full = p.full_bpp as usize;
    let w = width as usize;
    let Some(need_src) = full.checked_mul(w) else {
        return false;
    };
    let Some(need_dst) = plane_bpp.checked_mul(w) else {
        return false;
    };
    if src_row.len() < need_src || dst_plane.len() < need_dst {
        return false;
    }
    for x in 0..w {
        let t = &src_row[x * full..x * full + full];
        let d = &mut dst_plane[x * plane_bpp..x * plane_bpp + plane_bpp];
        if !extract_depth_stencil_plane(format, depth_aspect, stencil_aspect, t, d) {
            return false;
        }
    }
    true
}

/// Insert a tight plane row into a strided packed texture row (RMW per texel).
pub fn insert_plane_row(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    src_plane: &[u8],
    width: u32,
    dst_row: &mut [u8],
) -> bool {
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    let plane_bpp = if depth_aspect {
        p.depth_plane_bpp
    } else if stencil_aspect {
        p.stencil_plane_bpp
    } else {
        return false;
    } as usize;
    let full = p.full_bpp as usize;
    let w = width as usize;
    let Some(need_dst) = full.checked_mul(w) else {
        return false;
    };
    let Some(need_src) = plane_bpp.checked_mul(w) else {
        return false;
    };
    if dst_row.len() < need_dst || src_plane.len() < need_src {
        return false;
    }
    for x in 0..w {
        let s = &src_plane[x * plane_bpp..x * plane_bpp + plane_bpp];
        let t = &mut dst_row[x * full..x * full + full];
        if !insert_depth_stencil_plane(format, depth_aspect, stencil_aspect, s, t) {
            return false;
        }
    }
    true
}

/// Whether `format` stores sRGB-encoded values, so Metal decodes on sample and
/// encodes on write.
///
/// The class lookups below deliberately fold each `_SRGB` format onto its
/// linear sibling — the classes name a *byte layout*, and the two share one.
/// That fold is only safe because this predicate exists beside it: a caller that
/// takes the class has lost the qualifier and can ask here whether it just did.
/// Without it the loss is indistinguishable from the format never having been
/// sRGB at all.
pub fn is_srgb(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_RGBA8_UNORM_SRGB | MTL_FORMAT_BGRA8_UNORM_SRGB
    )
}

pub fn sampled_class(format: u16) -> Option<SampledClass> {
    Some(match format {
        MTL_FORMAT_A8_UNORM => SampledClass::A8Unorm,
        MTL_FORMAT_R8_UNORM => SampledClass::R8Unorm,
        MTL_FORMAT_RG8_UNORM => SampledClass::Rg8Unorm,
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => SampledClass::Rgba8Unorm,
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => SampledClass::Bgra8Unorm,
        MTL_FORMAT_RGBA16_FLOAT => SampledClass::Rgba16Float,
        _ => return None,
    })
}

/// Which storage-image selector a Metal format maps to, or `None` for a format
/// this host will not expose as a storage image.
///
/// The texel width is **not** returned. It used to be, as a second column
/// beside each selector, and the only thing that column was ever used for was a
/// `debug_assert_eq!` against [`bytes_per_pixel`] at three of the four call
/// sites — the fourth discarded it. A number stated twice that nothing reads is
/// a number that can disagree with itself in a release build, where a
/// `debug_assert` is not compiled at all. `storage_texel_width_matches_the_pixel_table`
/// now holds the same invariant for every `u16`, at test time, exhaustively.
pub fn storage_selector(format: u16) -> Option<StorageImageSelector> {
    Some(match format {
        MTL_FORMAT_R8_UNORM => StorageImageSelector::R8Unorm,
        MTL_FORMAT_R32_UINT => StorageImageSelector::R32Uint,
        MTL_FORMAT_RG8_UNORM => StorageImageSelector::Rg8Unorm,
        MTL_FORMAT_R16_FLOAT => StorageImageSelector::R16Float,
        MTL_FORMAT_RG16_FLOAT => StorageImageSelector::Rg16Float,
        MTL_FORMAT_RGBA8_UNORM => StorageImageSelector::Rgba8Unorm,
        MTL_FORMAT_BGRA8_UNORM => StorageImageSelector::Bgra8Unorm,
        MTL_FORMAT_RGBA8_UINT => StorageImageSelector::Rgba8Uint,
        MTL_FORMAT_RGBA8_SINT => StorageImageSelector::Rgba8Sint,
        MTL_FORMAT_RGBA16_UINT => StorageImageSelector::Rgba16Uint,
        MTL_FORMAT_RGBA16_FLOAT => StorageImageSelector::Rgba16Float,
        MTL_FORMAT_RGBA32_UINT => StorageImageSelector::Rgba32Uint,
        MTL_FORMAT_RGBA32_FLOAT => StorageImageSelector::Rgba32Float,
        _ => return None,
    })
}

/// Storage bytes per texel of a format this host will render into, or `None`
/// for one it will not.
///
/// The match arms *are* the renderable set — the answer to "may a colour
/// attachment be this format" is `.is_some()`, which is how
/// `runtime/metal_draw` asks it. There used to be a `RenderTargetClass` enum
/// returned alongside the width, one variant per arm below; every caller
/// discarded it, so it named the same six formats a second time and could
/// disagree with this list without anything noticing.
///
/// sRGB variants share storage bpp with their unorm counterparts (Metal texture
/// view rules).
pub fn render_target_bpp(format: u16) -> Option<u32> {
    Some(match format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => RGBA8_BPP,
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => BGRA8_BPP,
        MTL_FORMAT_RGBA16_FLOAT => RGBA16F_BPP,
        MTL_FORMAT_RG16_FLOAT => RG16F_BPP,
        MTL_FORMAT_R16_FLOAT => R16F_BPP,
        _ => return None,
    })
}

pub fn tight_row_bytes(width: u32, format: u16) -> Option<u32> {
    if width == 0 {
        return None;
    }
    let bpp = bytes_per_pixel(format)?;
    width.checked_mul(bpp)
}

pub fn swizzle_identity() -> SwizzlePlan {
    SwizzlePlan {
        source: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
    }
}

fn swizzle_selector_source(selector: u8) -> Option<SwizzleSource> {
    Some(match selector {
        0 => SwizzleSource::Zero,
        1 => SwizzleSource::One,
        2 => SwizzleSource::R,
        3 => SwizzleSource::G,
        4 => SwizzleSource::B,
        5 => SwizzleSource::A,
        _ => return None,
    })
}

pub fn swizzle_plan(raw: &[u8; COMPONENT_COUNT]) -> Option<SwizzlePlan> {
    let mut source = [SwizzleSource::Zero; COMPONENT_COUNT];
    for i in 0..COMPONENT_COUNT {
        source[i] = swizzle_selector_source(raw[i])?;
    }
    Some(SwizzlePlan { source })
}

pub fn swizzle_is_identity(plan: &SwizzlePlan) -> bool {
    plan.source
        == [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ]
}

pub fn apply_swizzle_rgba8(plan: &SwizzlePlan, in_rgba: [u8; 4]) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (component, source) in out.iter_mut().zip(plan.source) {
        *component = match source {
            SwizzleSource::Zero => UNORM8_MIN,
            SwizzleSource::One => UNORM8_MAX,
            SwizzleSource::R => in_rgba[COMPONENT_R],
            SwizzleSource::G => in_rgba[COMPONENT_G],
            SwizzleSource::B => in_rgba[COMPONENT_B],
            SwizzleSource::A => in_rgba[COMPONENT_A],
        };
    }
    out
}

pub fn f64_to_unorm8(value: f64) -> u8 {
    if !matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        UNORM8_MIN
    } else if value >= 1.0 {
        UNORM8_MAX
    } else {
        (value * f64::from(UNORM8_MAX) + 0.5) as u8
    }
}

pub fn f16_to_f32(half_bits: u16) -> f32 {
    let sign = (u32::from(half_bits & F16_SIGN_MASK)) << F16_F32_SIGN_SHIFT;
    let exp = (u32::from(half_bits) >> F16_EXP_SHIFT) & F16_EXP_MASK;
    let mut mant = u32::from(half_bits) & F16_MANT_MASK;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut normal_exp: i32 = 1;
            while (mant & F16_HIDDEN_BIT) == 0 {
                mant <<= 1;
                normal_exp -= 1;
            }
            mant &= F16_MANT_MASK;
            sign | (((normal_exp - F16_EXP_BIAS + F32_EXP_BIAS) as u32) << F32_EXP_SHIFT)
                | (mant << F16_F32_MANT_SHIFT)
        }
    } else if exp == F16_EXP_INF_NAN {
        sign | F32_INF_BITS | (mant << F16_F32_MANT_SHIFT)
    } else {
        let f32_exp = (exp as i32 - F16_EXP_BIAS + F32_EXP_BIAS) as u32;
        sign | (f32_exp << F32_EXP_SHIFT) | (mant << F16_F32_MANT_SHIFT)
    };
    f32::from_bits(bits)
}

fn build_f16_to_unorm8_lut() -> Box<[u8; 65536]> {
    let mut lut = Box::new([0u8; 65536]);
    for i in 0..=u16::MAX {
        let f = f16_to_f32(i);
        lut[i as usize] = if !matches!(f.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            UNORM8_MIN
        } else if f >= 1.0 {
            UNORM8_MAX
        } else {
            (f * f32::from(UNORM8_MAX) + 0.5) as u8
        };
    }
    lut
}

fn f16_to_unorm8_lut() -> &'static [u8; 65536] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Box<[u8; 65536]>> = OnceLock::new();
    LUT.get_or_init(build_f16_to_unorm8_lut)
}

fn unorm8_to_f16_slow(value: u8) -> u16 {
    let f = f32::from(value) / f32::from(UNORM8_MAX);
    let x = f.to_bits();
    let sign = ((x >> F16_F32_SIGN_SHIFT) as u16) & F16_SIGN_MASK;
    let e = ((x >> F32_EXP_SHIFT) & F32_EXP_MASK) as i32 - F32_EXP_BIAS + F16_EXP_BIAS;
    let mut m = x & F32_MANT_MASK;

    if f <= 0.0 {
        return sign;
    }
    if e >= F16_EXP_INF_NAN as i32 {
        return sign | F16_INF_BITS;
    }
    if e <= 0 {
        if e < F16_SUBNORMAL_EXP_MIN {
            return sign;
        }
        m |= F32_HIDDEN_BIT;
        let shift = (F16_SUBNORMAL_SHIFT_BASE - e) as u32;
        let mut hm = m >> shift;
        if ((m >> (shift - 1)) & 1) != 0 {
            hm += 1;
        }
        return sign | (hm as u16);
    }

    let mut h = sign | (((e as u32) << F16_EXP_SHIFT) as u16) | ((m >> F16_F32_MANT_SHIFT) as u16);
    if (m & F32_TO_F16_ROUND_BIT) != 0 {
        h = h.wrapping_add(1);
    }
    h
}

fn unorm8_to_f16_lut() -> &'static [u16; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u16; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0u16; 256];
        for i in 0..=UNORM8_MAX {
            lut[i as usize] = unorm8_to_f16_slow(i);
        }
        lut
    })
}

pub fn texel_to_rgba8(format: u16, src: &[u8]) -> Option<[u8; 4]> {
    let bpp = bytes_per_pixel(format)? as usize;
    if src.len() < bpp {
        return None;
    }
    let mut rgba = [0u8; 4];
    match format {
        MTL_FORMAT_A8_UNORM => {
            rgba[COMPONENT_A] = src[0];
        }
        MTL_FORMAT_R8_UNORM => {
            rgba[COMPONENT_R] = src[0];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_RG8_UNORM => {
            rgba[COMPONENT_R] = src[0];
            rgba[COMPONENT_G] = src[1];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            rgba.copy_from_slice(&src[..4]);
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {
            rgba[COMPONENT_R] = src[2];
            rgba[COMPONENT_G] = src[1];
            rgba[COMPONENT_B] = src[0];
            rgba[COMPONENT_A] = src[3];
        }
        MTL_FORMAT_RGBA16_FLOAT => {
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_G] = lut[ld16(&src[2..4]) as usize];
            rgba[COMPONENT_B] = lut[ld16(&src[4..6]) as usize];
            rgba[COMPONENT_A] = lut[ld16(&src[6..8]) as usize];
        }
        MTL_FORMAT_BGR10A2_UNORM => {
            // One little-endian 32-bit word: A in bits 31..30, then R, G, B in
            // descending 10-bit fields. That is the layout of the FourCC the
            // guest labels these surfaces with -- `l10r`, CoreVideo's
            // `30RGBLEPackedWideGamut` -- whose blue channel occupies the low
            // bits and whose red sits just under the alpha pair.
            //
            // Narrowed to 8 bits by taking the top 8 of each 10-bit field. That
            // is a truncation, not a rescale, and it is the same thing the 16F
            // LUT above does: this path exists to feed an RGBA8 sampled copy, so
            // the two low bits have nowhere to go. Wide-gamut content therefore
            // samples at 8-bit precision rather than not at all.
            let w = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
            let a2 = (w >> 30) & 0x3;
            rgba[COMPONENT_R] = (((w >> 20) & 0x3ff) >> 2) as u8;
            rgba[COMPONENT_G] = (((w >> 10) & 0x3ff) >> 2) as u8;
            rgba[COMPONENT_B] = ((w & 0x3ff) >> 2) as u8;
            // 2-bit alpha spans 0..3; replicate it across the byte so 3 is fully
            // opaque rather than 0xc0.
            rgba[COMPONENT_A] = (a2 * 0x55) as u8;
        }
        MTL_FORMAT_RGBA16_UNORM => {
            // Four 16-bit unorm channels, little-endian. Narrowed by taking each
            // channel's high byte -- the exact 8-bit unorm value, since unorm16
            // to unorm8 is a truncation of the low 8 bits.
            rgba[COMPONENT_R] = src[1];
            rgba[COMPONENT_G] = src[3];
            rgba[COMPONENT_B] = src[5];
            rgba[COMPONENT_A] = src[7];
        }
        MTL_FORMAT_RG16_FLOAT => {
            // Two float16 channels → R,G; B has no source (0), A opaque. Mirrors
            // the RGBA16Float LUT path (values clamp to [0,1] through the u8 LUT).
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_G] = lut[ld16(&src[2..4]) as usize];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_R16_FLOAT => {
            // One float16 channel -> R; G and B have no source (0), A opaque.
            // The single-channel sibling of the RG16Float arm above, and the
            // same LUT.
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        _ => return None,
    }
    Some(rgba)
}

pub fn rgba8_to_texel(format: u16, rgba: [u8; 4], dst: &mut [u8]) -> bool {
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    if dst.len() < bpp as usize {
        return false;
    }
    match format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            dst[..4].copy_from_slice(&rgba);
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {
            dst[0] = rgba[COMPONENT_B];
            dst[1] = rgba[COMPONENT_G];
            dst[2] = rgba[COMPONENT_R];
            dst[3] = rgba[COMPONENT_A];
        }
        MTL_FORMAT_RGBA16_FLOAT => {
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
            st16(&mut dst[2..4], lut[rgba[COMPONENT_G] as usize]);
            st16(&mut dst[4..6], lut[rgba[COMPONENT_B] as usize]);
            st16(&mut dst[6..8], lut[rgba[COMPONENT_A] as usize]);
        }
        MTL_FORMAT_RG16_FLOAT => {
            // R,G → two float16 channels; B,A have no destination (RG16 is
            // 4 bytes). Inverse of the texel_to_rgba8 RG16Float path.
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
            st16(&mut dst[2..4], lut[rgba[COMPONENT_G] as usize]);
        }
        MTL_FORMAT_R16_FLOAT => {
            // R -> one float16 channel; G,B,A have no destination. Inverse of
            // the texel_to_rgba8 R16Float path.
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
        }
        _ => return false,
    }
    true
}

fn row_walk_backward(
    src_len: usize,
    src_stride: usize,
    dst_len: usize,
    dst_stride: usize,
    same_base: bool,
) -> bool {
    // Non-overlapping or zero lengths: forward.
    if src_len == 0 || dst_len == 0 {
        return false;
    }
    // We cannot detect true pointer overlap without raw pointers; for Rust
    // slice APIs we only allow in-place when same_base is true (caller asserts
    // src and dst alias the same allocation).
    if !same_base {
        return false;
    }
    dst_stride > src_stride
}

pub fn convert_row_to_rgba8(format: u16, src: &[u8], pixels: u32, dst_rgba: &mut [u8]) -> bool {
    convert_row_to_rgba8_ex(format, src, pixels, dst_rgba, false)
}

fn convert_row_to_rgba8_ex(
    format: u16,
    src: &[u8],
    pixels: u32,
    dst_rgba: &mut [u8],
    same_base: bool,
) -> bool {
    if pixels == 0 {
        return true;
    }
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    let src_len = match (pixels as u64).checked_mul(bpp as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    let dst_len = match (pixels as u64).checked_mul(RGBA8_BPP as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    if src.len() < src_len || dst_rgba.len() < dst_len {
        return false;
    }
    let backward = row_walk_backward(
        src_len,
        bpp as usize,
        dst_len,
        RGBA8_BPP as usize,
        same_base,
    );

    if format == MTL_FORMAT_RGBA16_FLOAT {
        let lut = f16_to_unorm8_lut();
        let iter: Box<dyn Iterator<Item = u32>> = if backward {
            Box::new((0..pixels).rev())
        } else {
            Box::new(0..pixels)
        };
        for i in iter {
            let sp = (i as usize) * RGBA16F_BPP as usize;
            let dp = (i as usize) * RGBA8_BPP as usize;
            dst_rgba[dp + COMPONENT_R] = lut[ld16(&src[sp..sp + 2]) as usize];
            dst_rgba[dp + COMPONENT_G] = lut[ld16(&src[sp + 2..sp + 4]) as usize];
            dst_rgba[dp + COMPONENT_B] = lut[ld16(&src[sp + 4..sp + 6]) as usize];
            dst_rgba[dp + COMPONENT_A] = lut[ld16(&src[sp + 6..sp + 8]) as usize];
        }
        return true;
    }

    let iter: Box<dyn Iterator<Item = u32>> = if backward {
        Box::new((0..pixels).rev())
    } else {
        Box::new(0..pixels)
    };
    for i in iter {
        let sp = (i as usize) * bpp as usize;
        let dp = (i as usize) * RGBA8_BPP as usize;
        let Some(rgba) = texel_to_rgba8(format, &src[sp..sp + bpp as usize]) else {
            return false;
        };
        dst_rgba[dp..dp + 4].copy_from_slice(&rgba);
    }
    true
}

pub fn convert_rgba8_to_row(format: u16, src_rgba: &[u8], pixels: u32, dst: &mut [u8]) -> bool {
    if pixels == 0 {
        return true;
    }
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    let src_len = match (pixels as u64).checked_mul(RGBA8_BPP as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    let dst_len = match (pixels as u64).checked_mul(bpp as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    if src_rgba.len() < src_len || dst.len() < dst_len {
        return false;
    }

    if format == MTL_FORMAT_RGBA16_FLOAT {
        let lut = unorm8_to_f16_lut();
        for i in 0..pixels {
            let sp = (i as usize) * RGBA8_BPP as usize;
            let dp = (i as usize) * RGBA16F_BPP as usize;
            st16(
                &mut dst[dp..dp + 2],
                lut[src_rgba[sp + COMPONENT_R] as usize],
            );
            st16(
                &mut dst[dp + 2..dp + 4],
                lut[src_rgba[sp + COMPONENT_G] as usize],
            );
            st16(
                &mut dst[dp + 4..dp + 6],
                lut[src_rgba[sp + COMPONENT_B] as usize],
            );
            st16(
                &mut dst[dp + 6..dp + 8],
                lut[src_rgba[sp + COMPONENT_A] as usize],
            );
        }
        return true;
    }

    for i in 0..pixels {
        let sp = (i as usize) * RGBA8_BPP as usize;
        let dp = (i as usize) * bpp as usize;
        let mut rgba = [0u8; 4];
        rgba.copy_from_slice(&src_rgba[sp..sp + 4]);
        if !rgba8_to_texel(format, rgba, &mut dst[dp..dp + bpp as usize]) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BGR10A2Unorm` is pinned against the guest's own numbers, not a name.
    ///
    /// Tahoe's compositor draws window content in this format, and before it was
    /// modelled every such view was declined — 424 refusals in one measured boot,
    /// which is most of the on-screen text and chrome. The guest described those
    /// surfaces as FourCC `l10r` (`0x6c313072`, CoreVideo
    /// `30RGBLEPackedWideGamut`) with `w=365 bpr=1472` and `w=370 bpr=1536`:
    /// four bytes a texel, rounded up to 64. Both rows are asserted here so the
    /// size is tied to the measurement that established it.
    #[test]
    fn bgr10a2_is_four_bytes_and_unpacks_from_the_low_bits_up() {
        assert_eq!(bytes_per_pixel(MTL_FORMAT_BGR10A2_UNORM), Some(4));
        for (w, bpr) in [(365u32, 1472u32), (370, 1536)] {
            let tight = w * 4;
            assert!(bpr >= tight, "row must hold w*4");
            assert_eq!(bpr % 64, 0, "guest rows are 64-byte aligned");
            assert!(bpr - tight < 64, "and no more padding than that");
        }

        // Field order: A in 31..30, R in 29..20, G in 19..10, B in 9..0.
        // Full-scale red only.
        let red = (0x3u32 << 30) | (0x3ffu32 << 20);
        assert_eq!(
            texel_to_rgba8(MTL_FORMAT_BGR10A2_UNORM, &red.to_le_bytes()),
            Some([255, 0, 0, 255])
        );
        // Full-scale blue only — the low field, which is what distinguishes this
        // from an RGB10A2 reading.
        let blue = (0x3u32 << 30) | 0x3ff;
        assert_eq!(
            texel_to_rgba8(MTL_FORMAT_BGR10A2_UNORM, &blue.to_le_bytes()),
            Some([0, 0, 255, 255])
        );
        // Green, and a zero alpha that must stay zero rather than becoming opaque.
        let green_transparent = 0x3ffu32 << 10;
        assert_eq!(
            texel_to_rgba8(MTL_FORMAT_BGR10A2_UNORM, &green_transparent.to_le_bytes()),
            Some([0, 255, 0, 0])
        );
    }

    /// `RGBA16Unorm` is likewise pinned to the guest's allocation, and its
    /// narrowing to RGBA8 must take each channel's high byte.
    #[test]
    fn rgba16unorm_is_eight_bytes_and_narrows_by_high_byte() {
        assert_eq!(bytes_per_pixel(MTL_FORMAT_RGBA16_UNORM), Some(8));
        // Measured: a 384x384 view arrives with bpr=3072, alloc=1179648.
        assert_eq!(384 * 8, 3072);
        assert_eq!(384 * 384 * 8, 1179648);

        // 0xffff -> 255, 0x8080 -> 0x80, 0x00ff -> 0 (truncation, not rounding).
        let texel = [0xff, 0xff, 0x80, 0x80, 0xff, 0x00, 0x00, 0x40];
        assert_eq!(
            texel_to_rgba8(MTL_FORMAT_RGBA16_UNORM, &texel),
            Some([0xff, 0x80, 0x00, 0x40])
        );
    }

    /// A format the guest renders into must be render-targetable, and the
    /// admission tables must agree about it.
    ///
    /// `R16Float` was accepted by `bytes_per_pixel` and `storage_selector`, but
    /// not by the render-target table. One unresolvable colour attachment drops
    /// the whole render pass, so the disagreement costs every remaining draw in
    /// the serialized list.
    #[test]
    fn a_single_channel_float16_target_is_renderable_and_round_trips() {
        assert_eq!(render_target_bpp(MTL_FORMAT_R16_FLOAT), Some(R16F_BPP));
        assert_eq!(
            bytes_per_pixel(MTL_FORMAT_R16_FLOAT),
            render_target_bpp(MTL_FORMAT_R16_FLOAT)
        );

        for value in [0u8, 1, 64, 128, 200, 255] {
            let mut texel = [0u8; 2];
            assert!(rgba8_to_texel(
                MTL_FORMAT_R16_FLOAT,
                [value, 77, 88, 99],
                &mut texel
            ));
            let back =
                texel_to_rgba8(MTL_FORMAT_R16_FLOAT, &texel).expect("R16Float must decode");
            assert_eq!(back[COMPONENT_R], value, "R must survive at {value}");
            assert_eq!(back[COMPONENT_G], 0);
            assert_eq!(back[COMPONENT_B], 0);
            assert_eq!(back[COMPONENT_A], UNORM8_MAX);
        }

        assert_eq!(render_target_bpp(MTL_FORMAT_R8_UNORM), None);
    }

    #[test]
    fn bytes_per_pixel_matrix() {
        let cases = [
            (MTL_FORMAT_BGR10A2_UNORM, 4),
            (MTL_FORMAT_RGBA16_UNORM, 8),
            (MTL_FORMAT_A8_UNORM, 1),
            (MTL_FORMAT_R8_UNORM, 1),
            (MTL_FORMAT_R16_UNORM, 2),
            (MTL_FORMAT_R16_FLOAT, 2),
            (MTL_FORMAT_RG8_UNORM, 2),
            (MTL_FORMAT_R32_UINT, 4),
            (MTL_FORMAT_R32_SINT, 4),
            (MTL_FORMAT_R32_FLOAT, 4),
            (MTL_FORMAT_RG16_UNORM, 4),
            (MTL_FORMAT_RG16_UINT, 4),
            (MTL_FORMAT_RG16_FLOAT, 4),
            (MTL_FORMAT_RGBA8_UNORM, 4),
            (MTL_FORMAT_RGBA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGBA8_UINT, 4),
            (MTL_FORMAT_RGBA8_SINT, 4),
            (MTL_FORMAT_BGRA8_UNORM, 4),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGB9E5_FLOAT, 4),
            (MTL_FORMAT_RGBA16_UINT, 8),
            (MTL_FORMAT_RGBA16_FLOAT, 8),
            (MTL_FORMAT_RGBA32_UINT, 16),
            (MTL_FORMAT_RGBA32_FLOAT, 16),
            (MTL_FORMAT_DEPTH16_UNORM, 2),
            (MTL_FORMAT_DEPTH32_FLOAT, 4),
            (MTL_FORMAT_STENCIL8, 1),
            (MTL_FORMAT_DEPTH24_UNORM_STENCIL8, 4),
            (MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, 8),
            (MTL_FORMAT_X32_STENCIL8, 8),
            (MTL_FORMAT_X24_STENCIL8, 4),
        ];
        for (fmt, bpp) in cases {
            assert_eq!(bytes_per_pixel(fmt), Some(bpp));
        }
        assert_eq!(bytes_per_pixel(0xffff), None);
    }

    #[test]
    fn blit_aspect_bpp_depth_stencil() {
        // Pure depth + depth option.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, true, false),
            Some(4)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, false, false),
            Some(4)
        );
        // Pure depth cannot take stencil option.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, false, true),
            None
        );
        // Pure stencil.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_STENCIL8, false, true),
            Some(1)
        );
        // Combined: depth plane 4 B, stencil 1 B, full = packing full_bpp.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, true, false),
            Some(4)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, false, true),
            Some(1)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, false, false),
            Some(8)
        );
        assert!(blit_aspect_needs_repack(
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            true,
            false
        ));
        assert!(!blit_aspect_needs_repack(
            MTL_FORMAT_DEPTH32_FLOAT,
            true,
            false
        ));
        // Color cannot take DS options.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_BGRA8_UNORM, true, false),
            None
        );
    }

    /// Exactly the two `_SRGB` wire values carry the transfer function, and the
    /// class lookups fold each onto the linear sibling's byte layout. Both
    /// halves matter: the fold is what makes a class usable, `is_srgb` is what
    /// keeps the fold from being a silent loss.
    #[test]
    fn srgb_is_named_beside_the_class_that_folds_it() {
        assert!(is_srgb(MTL_FORMAT_RGBA8_UNORM_SRGB));
        assert!(is_srgb(MTL_FORMAT_BGRA8_UNORM_SRGB));
        for fmt in [
            MTL_FORMAT_RGBA8_UNORM,
            MTL_FORMAT_BGRA8_UNORM,
            MTL_FORMAT_A8_UNORM,
            MTL_FORMAT_RGBA16_FLOAT,
            MTL_FORMAT_DEPTH32_FLOAT,
            0xffff,
        ] {
            assert!(!is_srgb(fmt), "{fmt:#x}");
        }
        assert_eq!(
            sampled_class(MTL_FORMAT_RGBA8_UNORM_SRGB),
            sampled_class(MTL_FORMAT_RGBA8_UNORM)
        );
        assert_eq!(
            sampled_class(MTL_FORMAT_BGRA8_UNORM_SRGB),
            sampled_class(MTL_FORMAT_BGRA8_UNORM)
        );
        // The render-target rail does NOT fold the qualifier, and `is_srgb` is
        // where it is kept. `render_target_bpp` deliberately cannot say: a
        // storage width is the same eight bits either way, and the enum that
        // used to carry both the width and an sRGB-qualified variant name was
        // read by nobody, so the qualifier only ever came from here.
        assert!(is_srgb(MTL_FORMAT_RGBA8_UNORM_SRGB) != is_srgb(MTL_FORMAT_RGBA8_UNORM));
        assert_eq!(
            render_target_bpp(MTL_FORMAT_RGBA8_UNORM_SRGB),
            render_target_bpp(MTL_FORMAT_RGBA8_UNORM),
            "the sRGB qualifier does not change how wide a texel is"
        );
    }

    #[test]
    fn sampled_and_storage() {
        assert_eq!(
            sampled_class(MTL_FORMAT_A8_UNORM),
            Some(SampledClass::A8Unorm)
        );
        assert_eq!(sampled_class(MTL_FORMAT_R16_FLOAT), None);
        assert_eq!(
            storage_selector(MTL_FORMAT_R8_UNORM),
            Some(StorageImageSelector::R8Unorm)
        );
        assert_eq!(storage_selector(MTL_FORMAT_A8_UNORM), None);
        // R32Uint is storage-capable (specialized to the R32ui storage path);
        // its single-channel sint/float siblings are not.
        assert_eq!(
            storage_selector(MTL_FORMAT_R32_UINT),
            Some(StorageImageSelector::R32Uint)
        );
        assert_eq!(storage_selector(MTL_FORMAT_R32_SINT), None);
        assert_eq!(storage_selector(MTL_FORMAT_R32_FLOAT), None);
        assert_eq!(render_target_bpp(MTL_FORMAT_BGRA8_UNORM), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA8_UNORM), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA8_UNORM_SRGB), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_BGRA8_UNORM_SRGB), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA16_FLOAT), Some(8));
        assert_eq!(render_target_bpp(MTL_FORMAT_RG16_FLOAT), Some(4));
        // Integer / non-color formats stay fail-closed.
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA8_UINT), None);
        assert_eq!(render_target_bpp(MTL_FORMAT_R8_UNORM), None);
    }

    /// RG16Float MRT slots (vibrancy UI tile masks) must admit as color RTs so
    /// `mrt_draw_request` no longer drops the whole pass. Two channels survive
    /// the RGBA8-intermediate round trip; B has no source and A is opaque.
    #[test]
    fn rg16float_render_target_roundtrips_two_channels() {
        assert_eq!(render_target_bpp(MTL_FORMAT_RG16_FLOAT), Some(4));
        let w = 16u32;
        let mut rgba = vec![0u8; (w as usize) * 4];
        for i in 0..(w as usize) {
            rgba[i * 4] = 40; // R
            rgba[i * 4 + 1] = 90; // G
            rgba[i * 4 + 2] = 200; // B (dropped by RG16)
            rgba[i * 4 + 3] = 128; // A (dropped by RG16)
        }
        let tight = tight_row_bytes(w, MTL_FORMAT_RG16_FLOAT).unwrap();
        assert_eq!(tight, w * 4);
        let mut native = vec![0u8; tight as usize];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RG16_FLOAT,
            &rgba,
            w,
            &mut native
        ));
        let mut back = vec![0u8; (w as usize) * 4];
        assert!(convert_row_to_rgba8(
            MTL_FORMAT_RG16_FLOAT,
            &native,
            w,
            &mut back
        ));
        // R,G round-trip through the u8→f16→u8 LUT; B has no source (0); A opaque.
        assert_eq!(back[0], 40);
        assert_eq!(back[1], 90);
        assert_eq!(back[2], 0);
        assert_eq!(back[3], 255);
    }

    /// Metal color-renderable 8-bit + f16 set used as Reims VGPU pass attachments.
    /// Bring-up only admitted BGRA8/RGBA16F (compositor FBs); apps use RGBA8.
    #[test]
    fn color_renderable_formats_admit_app_rts() {
        for (fmt, bpp) in [
            (MTL_FORMAT_RGBA8_UNORM, 4u32),
            (MTL_FORMAT_RGBA8_UNORM_SRGB, 4),
            (MTL_FORMAT_BGRA8_UNORM, 4),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGBA16_FLOAT, 8),
        ] {
            assert_eq!(render_target_bpp(fmt), Some(bpp), "fmt={fmt:#x}");
            // Round-trip tight row for write_gva / mapping store.
            let w = 16u32;
            let mut rgba = vec![0u8; (w as usize) * 4];
            for i in 0..(w as usize) {
                rgba[i * 4] = 10;
                rgba[i * 4 + 1] = 20;
                rgba[i * 4 + 2] = 30;
                rgba[i * 4 + 3] = 255;
            }
            let tight = tight_row_bytes(w, fmt).unwrap();
            let mut native = vec![0u8; tight as usize];
            assert!(
                convert_rgba8_to_row(fmt, &rgba, w, &mut native),
                "convert host RGBA8 → guest fmt={fmt:#x}"
            );
            let mut back = vec![0u8; (w as usize) * 4];
            assert!(
                convert_row_to_rgba8(fmt, &native, w, &mut back),
                "convert guest fmt={fmt:#x} → host RGBA8"
            );
            // 8-bit unorm/sRGB and float16 (via unorm8 LUT) keep the solid color.
            assert_eq!(back[0], 10);
            assert_eq!(back[1], 20);
            assert_eq!(back[2], 30);
            assert_eq!(back[3], 255);
        }
    }

    /// The 128-byte-aligned IOSurface row lives in
    /// [`crate::contract::iosurface_pages::sample_window`], which is the one the
    /// mapper rail reads. A second `iosurface_row_bytes` here computed the same
    /// rule from its own copy of the alignment and served nothing but this test.
    #[test]
    fn rows_and_image_size() {
        use crate::contract::iosurface_pages::sample_window;
        let bpr = |w, fmt| sample_window(0, fmt, w, 1).map(|(_, bpr, _)| bpr);
        assert_eq!(bpr(200, MTL_FORMAT_BGRA8_UNORM), Some(896));
        assert_eq!(bpr(64, MTL_FORMAT_BGRA8_UNORM), Some(256));
        assert_eq!(bpr(250, MTL_FORMAT_BGRA8_UNORM), Some(1024));
        assert_eq!(bpr(200, MTL_FORMAT_RGBA16_FLOAT), Some(1664));
        assert_eq!(bpr(0, MTL_FORMAT_BGRA8_UNORM), None);
        // Same 4 Bpp packing as BGRA8 → same 128 B aligned row for w=200.
        assert_eq!(bpr(200, MTL_FORMAT_RGBA8_UNORM), Some(896));
        assert_eq!(tight_row_bytes(200, MTL_FORMAT_BGRA8_UNORM), Some(800));
    }

    #[test]
    fn swizzle_and_texels() {
        let plan = swizzle_plan(&[2, 3, 4, 5]).unwrap();
        assert!(swizzle_is_identity(&plan));
        let bgra = [10u8, 20, 30, 40];
        let rgba = texel_to_rgba8(MTL_FORMAT_BGRA8_UNORM, &bgra).unwrap();
        assert_eq!(rgba, [30, 20, 10, 40]);
        let mut out = [0u8; 4];
        assert!(rgba8_to_texel(MTL_FORMAT_BGRA8_UNORM, rgba, &mut out));
        assert_eq!(out, bgra);

        let row = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut rgba_row = [0u8; 8];
        assert!(convert_row_to_rgba8(
            MTL_FORMAT_RGBA8_UNORM,
            &row,
            2,
            &mut rgba_row
        ));
        assert_eq!(rgba_row, row);
        let mut back = [0u8; 8];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RGBA8_UNORM,
            &rgba_row,
            2,
            &mut back
        ));
        assert_eq!(back, row);

        // Property: the two lookup tables the conversion paths index are
        // exact inverses over every byte. Asserted against the tables
        // themselves, which is what those paths read.
        let to_f16 = unorm8_to_f16_lut();
        let to_u8 = f16_to_unorm8_lut();
        for v in 0u8..=255 {
            assert_eq!(to_u8[to_f16[v as usize] as usize], v);
        }
        let _ = f64_to_unorm8(0.5);
        let _ = f16_to_f32(0x3c00); // 1.0
    }

    /// Every storage-capable format has a texel width, and it is the width the
    /// selector table used to carry beside it.
    ///
    /// This replaces three `debug_assert_eq!(selector_bpp, bpp)` at the
    /// `compute_exec` staging call sites. Those checked the same thing, but only
    /// for formats a running guest happened to bind, and only in a debug build —
    /// a release build compiled them out entirely, so the disagreement they
    /// guarded against would have shipped silently. Sweeping the whole `u16`
    /// space costs microseconds and cannot miss an arm.
    ///
    /// The widths are spelled out rather than derived, because that is the point:
    /// a change to `bytes_per_pixel` that silently redefined a storage format's
    /// stride is exactly what the deleted column existed to catch.
    #[test]
    fn storage_texel_width_matches_the_pixel_table() {
        let expected: &[(u16, u32)] = &[
            (MTL_FORMAT_R8_UNORM, R8_BPP),
            (MTL_FORMAT_R32_UINT, R32_BPP),
            (MTL_FORMAT_RG8_UNORM, RG8_BPP),
            (MTL_FORMAT_R16_FLOAT, R16F_BPP),
            (MTL_FORMAT_RG16_FLOAT, RG16F_BPP),
            (MTL_FORMAT_RGBA8_UNORM, RGBA8_BPP),
            (MTL_FORMAT_BGRA8_UNORM, BGRA8_BPP),
            (MTL_FORMAT_RGBA8_UINT, RGBA8_BPP),
            (MTL_FORMAT_RGBA8_SINT, RGBA8_BPP),
            (MTL_FORMAT_RGBA16_UINT, RGBA16_BPP),
            (MTL_FORMAT_RGBA16_FLOAT, RGBA16F_BPP),
            (MTL_FORMAT_RGBA32_UINT, RGBA32_BPP),
            (MTL_FORMAT_RGBA32_FLOAT, RGBA32F_BPP),
        ];
        for (fmt, bpp) in expected {
            assert!(
                storage_selector(*fmt).is_some(),
                "format {fmt:#x} lost its storage selector"
            );
            assert_eq!(
                bytes_per_pixel(*fmt),
                Some(*bpp),
                "storage format {fmt:#x} changed texel width"
            );
        }

        // And no arm was added to one table without the other. Exhaustive, so
        // the two lists cannot drift apart in either direction.
        for fmt in 0u16..=u16::MAX {
            if storage_selector(fmt).is_some() {
                assert!(
                    bytes_per_pixel(fmt).is_some(),
                    "storage-capable {fmt:#x} has no texel width"
                );
                assert!(
                    expected.iter().any(|(f, _)| *f == fmt),
                    "storage-capable {fmt:#x} is not pinned above"
                );
            }
        }
    }

    /// A [`TexelLayout`]'s width is the width of the guest format it stands for.
    ///
    /// The layout enum and the `u16` format table are two vocabularies for one
    /// fact, and every sampled rail crosses between them: the format decides
    /// what the guest wrote, the layout decides the stride the host reads it
    /// at. Nothing compared them, and a disagreement is not a decode error —
    /// it is rows read at the wrong stride, which produces a sheared image
    /// rather than a refusal.
    ///
    /// The `match` is exhaustive on purpose. A new layout cannot be added
    /// without naming the guest format it represents, which is the question a
    /// new variant most needs to answer.
    #[test]
    fn a_texel_layout_is_as_wide_as_the_guest_format_it_stands_for() {
        for layout in [
            TexelLayout::Rgba8,
            TexelLayout::Bgra8,
            TexelLayout::R8,
            TexelLayout::Rg8,
            TexelLayout::R16Unorm,
            TexelLayout::Rg16Unorm,
            TexelLayout::Rg16Uint,
            TexelLayout::R16Float,
            TexelLayout::R32Float,
        ] {
            let mtl = match layout {
                TexelLayout::Rgba8 => MTL_FORMAT_RGBA8_UNORM,
                TexelLayout::Bgra8 => MTL_FORMAT_BGRA8_UNORM,
                TexelLayout::R8 => MTL_FORMAT_R8_UNORM,
                TexelLayout::Rg8 => MTL_FORMAT_RG8_UNORM,
                TexelLayout::R16Unorm => MTL_FORMAT_R16_UNORM,
                TexelLayout::Rg16Unorm => MTL_FORMAT_RG16_UNORM,
                TexelLayout::Rg16Uint => MTL_FORMAT_RG16_UINT,
                TexelLayout::R16Float => MTL_FORMAT_R16_FLOAT,
                TexelLayout::R32Float => MTL_FORMAT_R32_FLOAT,
            };
            assert_eq!(
                Some(layout.bytes_per_texel()),
                bytes_per_pixel(mtl),
                "{layout:?} and its guest format {mtl:#x} disagree on texel width"
            );
            // Exactly the byte-order layouts have a CPU loader class. The
            // native sampled/storage layouts deliberately do not: converting
            // them through the 8-bit colour LUTs would quantize floats and
            // reinterpret integer payloads. Pinned here because it is an
            // *absence*, and an absence is what a later "just add the missing
            // arm" edit removes without noticing what it was for.
            let expects_loader = matches!(
                layout,
                TexelLayout::Rgba8 | TexelLayout::Bgra8 | TexelLayout::R8 | TexelLayout::Rg8
            );
            assert_eq!(
                sampled_class(mtl).is_some(),
                expects_loader,
                "{layout:?} ({mtl:#x}) changed whether the CPU loader claims it"
            );
        }
    }

    /// Four bytes wide is not the same as a four-byte colour order.
    ///
    /// [`TexelLayout::is_four_byte_color`] gates the RGBA8-shaped loaders, the
    /// tight-row gathers and the zero-copy rails, all of which reinterpret the
    /// four bytes as R,G,B,A. `R32Float` is exactly as wide and is a single
    /// channel, so admitting it on width would hand a float LUT to a loader
    /// that swizzles it as colour. The enum's own doc states this; nothing
    /// checked it.
    #[test]
    fn four_bytes_wide_does_not_admit_a_layout_to_the_colour_rails() {
        for layout in [TexelLayout::Rgba8, TexelLayout::Bgra8] {
            assert!(layout.is_four_byte_color(), "{layout:?} is a colour order");
            assert_eq!(layout.bytes_per_texel(), RGBA8_BPP);
        }
        assert_eq!(TexelLayout::R32Float.bytes_per_texel(), RGBA8_BPP);
        assert!(
            !TexelLayout::R32Float.is_four_byte_color(),
            "a single-channel float is four bytes wide and is not a colour order"
        );
        for layout in [TexelLayout::R8, TexelLayout::Rg8, TexelLayout::R16Float] {
            assert!(!layout.is_four_byte_color());
            assert_ne!(layout.bytes_per_texel(), RGBA8_BPP);
        }
    }

    #[test]
    fn unsupported_fail_closed() {
        // Unknown formats fail closed. Depth/stencil families have bpp for blit
        // packing but remain unsampled / non-storage / non-RT.
        for fmt in [0xffffu16, 130, 204] {
            assert!(bytes_per_pixel(fmt).is_none());
            assert!(sampled_class(fmt).is_none());
            assert!(storage_selector(fmt).is_none());
            assert!(render_target_bpp(fmt).is_none());
            assert!(texel_to_rgba8(fmt, &[0; 16]).is_none());
        }
        for fmt in [
            MTL_FORMAT_DEPTH32_FLOAT,
            MTL_FORMAT_STENCIL8,
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            MTL_FORMAT_DEPTH24_UNORM_STENCIL8,
        ] {
            assert!(bytes_per_pixel(fmt).is_some());
            assert!(sampled_class(fmt).is_none());
            assert!(storage_selector(fmt).is_none());
            assert!(render_target_bpp(fmt).is_none());
            assert!(texel_to_rgba8(fmt, &[0; 16]).is_none());
        }
    }

    #[test]
    fn depth32_stencil8_plane_roundtrip() {
        let fmt = MTL_FORMAT_DEPTH32_FLOAT_STENCIL8;
        let p = depth_stencil_packing(fmt).unwrap();
        assert_eq!(p.full_bpp, 8);
        // Depth = 1.0f32, stencil = 0xAB
        let mut texel = [0u8; 8];
        texel[0..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        texel[4] = 0xab;
        let mut depth = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt, true, false, &texel, &mut depth
        ));
        assert_eq!(depth, 1.0f32.to_bits().to_le_bytes());
        let mut st = [0u8; 1];
        assert!(extract_depth_stencil_plane(
            fmt, false, true, &texel, &mut st
        ));
        assert_eq!(st[0], 0xab);
        // Insert new depth, keep stencil.
        let mut t2 = texel;
        let new_d = 0.5f32.to_bits().to_le_bytes();
        assert!(insert_depth_stencil_plane(
            fmt, true, false, &new_d, &mut t2
        ));
        assert_eq!(t2[4], 0xab);
        let mut d2 = [0u8; 4];
        assert!(extract_depth_stencil_plane(fmt, true, false, &t2, &mut d2));
        assert_eq!(d2, new_d);
        // Row extract 2 pixels.
        let mut row = [0u8; 16];
        row[..8].copy_from_slice(&texel);
        row[8..16].copy_from_slice(&t2);
        let mut planes = [0u8; 8];
        assert!(extract_plane_row(fmt, true, false, &row, 2, &mut planes));
        assert_eq!(&planes[0..4], &1.0f32.to_bits().to_le_bytes());
        assert_eq!(&planes[4..8], &new_d);
    }

    #[test]
    fn depth24_stencil8_plane_pack() {
        let fmt = MTL_FORMAT_DEPTH24_UNORM_STENCIL8;
        // stencil=0x11, depth24=0xAABBCC → packed LE
        let depth24 = 0x00aabbccu32;
        let packed = 0x11u32 | (depth24 << 8);
        let texel = packed.to_le_bytes();
        let mut depth = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt, true, false, &texel, &mut depth
        ));
        assert_eq!(u32::from_le_bytes(depth), depth24);
        let mut st = [0u8; 1];
        assert!(extract_depth_stencil_plane(
            fmt, false, true, &texel, &mut st
        ));
        assert_eq!(st[0], 0x11);
        let mut t2 = [0u8; 4];
        assert!(insert_depth_stencil_plane(
            fmt,
            false,
            true,
            &[0x22],
            &mut t2
        ));
        assert!(insert_depth_stencil_plane(
            fmt,
            true,
            false,
            &depth24.to_le_bytes(),
            &mut t2
        ));
        assert_eq!(u32::from_le_bytes(t2), 0x22 | (depth24 << 8));
    }

    #[test]
    fn property_fuzz_row_roundtrip_rgba8() {
        // Corpus-driven property: random-ish patterns through BGRA/RGBA convert.
        let patterns: &[&[u8]] = &[
            &[0, 0, 0, 0],
            &[255, 255, 255, 255],
            &[1, 2, 3, 4],
            &[10, 20, 30, 40, 50, 60, 70, 80],
            &[0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe],
        ];
        for p in patterns {
            let pixels = (p.len() / 4) as u32;
            if pixels == 0 {
                continue;
            }
            let mut rgba = vec![0u8; p.len()];
            assert!(convert_row_to_rgba8(
                MTL_FORMAT_RGBA8_UNORM,
                p,
                pixels,
                &mut rgba
            ));
            let mut back = vec![0u8; p.len()];
            assert!(convert_rgba8_to_row(
                MTL_FORMAT_RGBA8_UNORM,
                &rgba,
                pixels,
                &mut back
            ));
            assert_eq!(&back[..], *p);
        }
    }
}
