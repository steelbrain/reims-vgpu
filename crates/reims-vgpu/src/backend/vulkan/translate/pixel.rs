//! `MTLPixelFormat` → `VkFormat`, including the sRGB transfer function and the
//! component mapping Vulkan needs where its channel set differs from Metal's.
//!
//! # Why this table is total
//!
//! Before it, the same Metal→Vulkan pixel decision was re-made at each call
//! site that needed one, and the sRGB qualifier was folded into its linear
//! sibling at twelve independent sites with no record that anything was lost.
//! A lost qualifier looked exactly like a supported format. Here every
//! contract-defined `MTL_FORMAT_*` value has exactly one arm, `*_SRGB` formats
//! reach their `VK_FORMAT_*_SRGB` counterpart, and anything else declines by
//! name through [`TranslateReason::UnknownPixelFormat`].
//!
//! # sRGB is a choice, not an accident
//!
//! A path that genuinely cannot apply the transfer function (because it is
//! moving raw texels, not shading) asks for [`PixelFormat::linear_vk`] and
//! records the [`TranslateReason::SrgbDowngraded`] that
//! [`srgb_decline`] hands back. The loss is then one grep away instead
//! of invisible.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::StorageImageFormat;
use crate::contract::pixel_format::{
    self, SampledByteFormat, StorageImageSelector, SwizzlePlan, SwizzleSource, TexelLayout,
    COMPONENT_A, COMPONENT_B, COMPONENT_G, COMPONENT_R,
};

/// Whether a format's stored values carry the sRGB electro-optical transfer
/// function, which the hardware applies on sample and reverses on write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferFunction {
    Linear,
    Srgb,
}

/// One decoded Metal pixel format, expressed in Vulkan terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelFormat {
    /// The Vulkan format that reproduces the Metal format faithfully — sRGB
    /// included. This is what a render target or sampled image should bind
    /// unless something concrete prevents it.
    pub vk: vk::Format,
    /// Same channel order, same bit layout, linear transfer function. Equals
    /// [`Self::vk`] for a format that was already linear. Binding this for an
    /// sRGB Metal format is a **downgrade** and its call site must record
    /// [`TranslateReason::SrgbDowngraded`].
    pub linear_vk: vk::Format,
    pub transfer: TransferFunction,
    /// Bytes per texel in guest linear storage, from the decode contract — the
    /// single source, so the two can never drift.
    pub bytes_per_texel: u32,
    /// How Metal's presented `(r,g,b,a)` channels sit on this Vulkan format's
    /// channels. Identity for every format whose channel set Vulkan matches;
    /// non-identity only where the Vulkan 1.2 baseline has no equivalent
    /// format (see `A8Unorm`).
    pub components: SwizzlePlan,
}

impl PixelFormat {
    pub fn is_srgb(&self) -> bool {
        matches!(self.transfer, TransferFunction::Srgb)
    }
}

const IDENTITY: SwizzlePlan = SwizzlePlan {
    source: [
        SwizzleSource::R,
        SwizzleSource::G,
        SwizzleSource::B,
        SwizzleSource::A,
    ],
};

/// Metal `A8Unorm` presents `(0, 0, 0, a)`. The Vulkan 1.2 baseline has no
/// single-channel alpha format — `VK_FORMAT_A8_UNORM_KHR` arrived with
/// `VK_KHR_maintenance5`, well above the floor every matrix cell must meet — so
/// the byte rides in `R8_UNORM` and this mapping puts it back in alpha with the
/// colour channels zeroed. Identical to what the CPU texel path already
/// produces for this format.
const ALPHA_IN_RED: SwizzlePlan = SwizzlePlan {
    source: [
        SwizzleSource::Zero,
        SwizzleSource::Zero,
        SwizzleSource::Zero,
        SwizzleSource::R,
    ],
};

fn linear(vk: vk::Format, bytes_per_texel: u32) -> PixelFormat {
    PixelFormat {
        vk,
        linear_vk: vk,
        transfer: TransferFunction::Linear,
        bytes_per_texel,
        components: IDENTITY,
    }
}

fn srgb(vk: vk::Format, linear_vk: vk::Format, bytes_per_texel: u32) -> PixelFormat {
    PixelFormat {
        vk,
        linear_vk,
        transfer: TransferFunction::Srgb,
        bytes_per_texel,
        components: IDENTITY,
    }
}

/// Translate one decoded `MTLPixelFormat`.
///
/// Total over the values `crate::contract::pixel_format` defines; every other
/// value declines by name rather than reaching a default. Depth/stencil arms
/// are included because the same enum carries them on the wire — whether a
/// given role (colour attachment, storage image, sampled) admits a format is a
/// *contract* question answered by `render_target_bpp` / `storage_selector` /
/// `sampled_class`, and a *device* question answered by the capability layer;
/// neither is this function's job.
pub fn translate(mtl: u16) -> Result<PixelFormat, TranslateReason> {
    use pixel_format as p;
    Ok(match mtl {
        p::MTL_FORMAT_A8_UNORM => PixelFormat {
            components: ALPHA_IN_RED,
            ..linear(vk::Format::R8_UNORM, 1)
        },
        p::MTL_FORMAT_R8_UNORM => linear(vk::Format::R8_UNORM, 1),
        p::MTL_FORMAT_R8_UINT => linear(vk::Format::R8_UINT, 1),
        p::MTL_FORMAT_R16_UNORM => linear(vk::Format::R16_UNORM, 2),
        p::MTL_FORMAT_R16_FLOAT => linear(vk::Format::R16_SFLOAT, 2),
        p::MTL_FORMAT_RG8_UNORM => linear(vk::Format::R8G8_UNORM, 2),
        p::MTL_FORMAT_RG8_UINT => linear(vk::Format::R8G8_UINT, 2),
        p::MTL_FORMAT_RG16_UNORM => linear(vk::Format::R16G16_UNORM, 4),
        p::MTL_FORMAT_R32_UINT => linear(vk::Format::R32_UINT, 4),
        p::MTL_FORMAT_R32_SINT => linear(vk::Format::R32_SINT, 4),
        p::MTL_FORMAT_R32_FLOAT => linear(vk::Format::R32_SFLOAT, 4),
        p::MTL_FORMAT_RG16_FLOAT => linear(vk::Format::R16G16_SFLOAT, 4),
        p::MTL_FORMAT_RGBA8_UNORM => linear(vk::Format::R8G8B8A8_UNORM, 4),
        p::MTL_FORMAT_RGBA8_UNORM_SRGB => {
            srgb(vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM, 4)
        }
        p::MTL_FORMAT_RGBA8_UINT => linear(vk::Format::R8G8B8A8_UINT, 4),
        p::MTL_FORMAT_RGBA8_SINT => linear(vk::Format::R8G8B8A8_SINT, 4),
        p::MTL_FORMAT_BGRA8_UNORM => linear(vk::Format::B8G8R8A8_UNORM, 4),
        p::MTL_FORMAT_BGRA8_UNORM_SRGB => {
            srgb(vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM, 4)
        }
        p::MTL_FORMAT_RGB9E5_FLOAT => linear(vk::Format::E5B9G9R9_UFLOAT_PACK32, 4),
        // The BC block-compressed families. `bytes_per_texel` here is bytes per
        // **4x4 block** — 8 or 16 — which is what `pixel_format::block_geometry`
        // says and what every sizing expression on the sampled rail asks for.
        // The uncompressed arms above are the same field with a 1x1 block, so
        // this is not a second meaning; it is the same number with the grid
        // stated. See `pixel_format::MTL_FORMAT_BC1_RGBA` for why the family
        // arrives whole and `caps::device_features::DeviceFeatures::
        // texture_compression_bc` for the one feature that gates all of it.
        p::MTL_FORMAT_BC1_RGBA => linear(vk::Format::BC1_RGBA_UNORM_BLOCK, p::BC_BLOCK_BYTES_8),
        p::MTL_FORMAT_BC1_RGBA_SRGB => srgb(
            vk::Format::BC1_RGBA_SRGB_BLOCK,
            vk::Format::BC1_RGBA_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
        ),
        p::MTL_FORMAT_BC2_RGBA => linear(vk::Format::BC2_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC2_RGBA_SRGB => srgb(
            vk::Format::BC2_SRGB_BLOCK,
            vk::Format::BC2_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
        ),
        p::MTL_FORMAT_BC3_RGBA => linear(vk::Format::BC3_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC3_RGBA_SRGB => srgb(
            vk::Format::BC3_SRGB_BLOCK,
            vk::Format::BC3_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
        ),
        p::MTL_FORMAT_BC4_R_UNORM => linear(vk::Format::BC4_UNORM_BLOCK, p::BC_BLOCK_BYTES_8),
        p::MTL_FORMAT_BC4_R_SNORM => linear(vk::Format::BC4_SNORM_BLOCK, p::BC_BLOCK_BYTES_8),
        p::MTL_FORMAT_BC5_RG_UNORM => linear(vk::Format::BC5_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC5_RG_SNORM => linear(vk::Format::BC5_SNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC6H_RGB_FLOAT => {
            linear(vk::Format::BC6H_SFLOAT_BLOCK, p::BC_BLOCK_BYTES_16)
        }
        p::MTL_FORMAT_BC6H_RGB_UFLOAT => {
            linear(vk::Format::BC6H_UFLOAT_BLOCK, p::BC_BLOCK_BYTES_16)
        }
        p::MTL_FORMAT_BC7_RGBA_UNORM => linear(vk::Format::BC7_UNORM_BLOCK, p::BC_BLOCK_BYTES_16),
        p::MTL_FORMAT_BC7_RGBA_UNORM_SRGB => srgb(
            vk::Format::BC7_SRGB_BLOCK,
            vk::Format::BC7_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
        ),
        // The packed 32-bit colour family. Each Vulkan spelling is the same
        // word cut the same way as its Metal one — `A2B10G10R10` puts red in
        // the low bits as `RGB10A2Unorm` does, `A2R10G10B10` puts blue there as
        // `BGR10A2Unorm` does, and `B10G11R11` is `RG11B10Float`'s word — so a
        // guest texel is sampled unchanged rather than converted.
        p::MTL_FORMAT_RGB10A2_UNORM => linear(vk::Format::A2B10G10R10_UNORM_PACK32, 4),
        p::MTL_FORMAT_BGR10A2_UNORM => linear(vk::Format::A2R10G10B10_UNORM_PACK32, 4),
        p::MTL_FORMAT_RG11B10_FLOAT => linear(vk::Format::B10G11R11_UFLOAT_PACK32, 4),
        // `RGB10A2Uint` has no arm on purpose: an integer texel must not run
        // through the unorm converters, so it is declared for its width in the
        // decode contract and refused by name here, as `R8Uint` and `RG8Uint`
        // are.
        p::MTL_FORMAT_RGBA16_UNORM => linear(vk::Format::R16G16B16A16_UNORM, 8),
        p::MTL_FORMAT_RGBA16_UINT => linear(vk::Format::R16G16B16A16_UINT, 8),
        p::MTL_FORMAT_RGBA16_FLOAT => linear(vk::Format::R16G16B16A16_SFLOAT, 8),
        p::MTL_FORMAT_RGBA32_UINT => linear(vk::Format::R32G32B32A32_UINT, 16),
        p::MTL_FORMAT_RGBA32_FLOAT => linear(vk::Format::R32G32B32A32_SFLOAT, 16),
        p::MTL_FORMAT_DEPTH16_UNORM => linear(vk::Format::D16_UNORM, 2),
        p::MTL_FORMAT_DEPTH32_FLOAT => linear(vk::Format::D32_SFLOAT, 4),
        p::MTL_FORMAT_STENCIL8 => linear(vk::Format::S8_UINT, 1),
        p::MTL_FORMAT_DEPTH24_UNORM_STENCIL8 => linear(vk::Format::D24_UNORM_S8_UINT, 4),
        p::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 => linear(vk::Format::D32_SFLOAT_S8_UINT, 8),
        // Metal's `X*_Stencil8` are stencil-only *views* of the combined
        // depth-stencil cell, not distinct storage: the decode contract already
        // gives them the same cell size and stencil offset as the format they
        // view (`depth_stencil_packing`). Vulkan has no stencil-only view
        // format either, so they translate to the combined format and the
        // STENCIL aspect selects the plane. This is the contract's own layout,
        // not an invented fallback.
        p::MTL_FORMAT_X32_STENCIL8 => linear(vk::Format::D32_SFLOAT_S8_UINT, 8),
        p::MTL_FORMAT_X24_STENCIL8 => linear(vk::Format::D24_UNORM_S8_UINT, 4),
        other => return Err(TranslateReason::UnknownPixelFormat(other)),
    })
}

/// Whether a decoded Metal pixel format carries the sRGB transfer function.
///
/// Delegates to the decode contract so the crate has exactly one answer, and a
/// unit test holds the two in agreement.
pub fn is_srgb(mtl: u16) -> bool {
    pixel_format::is_srgb(mtl)
}

/// The guest texel layout for a decoded Metal pixel format, and the decline to
/// record if reaching it dropped the sRGB qualifier.
///
/// `Ok((layout, Some(reason)))` means the layout is right but the transfer
/// function was lost; `Ok((layout, None))` means nothing was lost. A format the
/// contract defines but this rail carries no layout for declines with
/// [`TranslateReason::NoSampledLayout`] — a *different* slug from an undefined
/// wire value, so the fail log distinguishes "we do not know this format" from
/// "this rail does not carry it".
///
/// The answer is a contract [`TexelLayout`], not a Vulkan format, because its
/// callers are the CPU-upload and in-place-gather rails in `runtime/`: they
/// reason about how many bytes a guest texel occupies and in which channel
/// order, which is a decode question. The host spelling of that layout is
/// [`vk_texel_layout`], applied once where the engine builds the image.
///
/// Callers still choose which layouts they accept: a rail that only handles
/// four-byte texels asks [`TexelLayout::is_four_byte_color`] rather than a
/// narrower entry point, so this table stays the single Metal-side rule.
pub fn sampled_pixels(
    mtl: u16,
) -> Result<(TexelLayout, Option<TranslateReason>, SwizzlePlan), TranslateReason> {
    let f = translate(mtl)?;
    // The compressed families answer from the contract rather than from a
    // `linear_vk` arm here, because `runtime::draw::texture_view` needs the same
    // answer and cannot reach this module. One mapping, asked twice — see
    // `pixel_format::block_compressed_layout`. Whether this host can sample it
    // is a capability the rail carries, not a fact of the translation.
    if let Some(layout) = pixel_format::block_compressed_layout(mtl) {
        return Ok((layout, None, pixel_format::swizzle_identity()));
    }
    // A format whose Metal channels do not sit identically on its Vulkan
    // channels needs a component mapping on the view to sample correctly.
    let layout = match f.linear_vk {
        vk::Format::R8G8B8A8_UNORM => TexelLayout::Rgba8,
        vk::Format::B8G8R8A8_UNORM => TexelLayout::Bgra8,
        vk::Format::R8_UNORM => TexelLayout::R8,
        vk::Format::R8G8_UNORM => TexelLayout::Rg8,
        // Single-channel float rides its own native rail (color-management
        // LUTs). `R16_SFLOAT` is a spec-mandatory sampled+linear format, so it
        // is unconditional. `R32_SFLOAT`'s linear-filter feature is optional
        // (absent on Apple/MoltenVK): the layout is named here (a decode fact),
        // but the rail that emits it must confirm the host can filter it — see
        // `try_linear_sample_zero_copy`'s `supports_sampled_r32f_linear_filter`
        // gate — or the sample stays fail-visible.
        vk::Format::R16_SFLOAT => TexelLayout::R16Float,
        vk::Format::R32_SFLOAT => TexelLayout::R32Float,
        // The ten-bit biplanar video planes, native for the reason the float
        // layouts above are native. Both are Vulkan-mandatory sampled formats
        // with mandatory linear filtering, so neither needs a capability gate.
        vk::Format::R16_UNORM => TexelLayout::R16Unorm,
        vk::Format::R16G16_UNORM => TexelLayout::Rg16Unorm,
        // The half-float colour layouts. A recent macOS window server
        // composites in `MTLPixelFormatRGBA16Float`, and every such bind used to
        // land on the CPU re-read rung and be quantized to unorm8 on the way in
        // — 99 % of this rail's format declines on a driven macos-26 boot. Both
        // are exact as guest bytes: the Metal and Vulkan spellings are the same
        // little-endian binary16 channels in the same order.
        vk::Format::R16G16B16A16_UNORM => TexelLayout::Rgba16Unorm,
        vk::Format::R16G16B16A16_SFLOAT => TexelLayout::Rgba16Float,
        vk::Format::R16G16_SFLOAT => TexelLayout::Rg16Float,
        // The packed 32-bit colour layouts, native for the reason the wide
        // layouts above are: the word is the guest's, bit for bit, and no
        // conversion to unorm8 could cut a ten- or eleven-bit channel without
        // discarding what the guest picked the format for.
        vk::Format::A2B10G10R10_UNORM_PACK32 => TexelLayout::Rgb10a2Unorm,
        vk::Format::A2R10G10B10_UNORM_PACK32 => TexelLayout::Bgr10a2Unorm,
        vk::Format::B10G11R11_UFLOAT_PACK32 => TexelLayout::Rg11b10Float,
        _ => return Err(TranslateReason::NoSampledLayout(mtl)),
    };
    // The format's own channel plan travels with the layout instead of being a
    // reason to refuse. A byte layout says how wide a texel is and in what
    // order its bytes sit; it cannot say that `A8Unorm`'s byte belongs in alpha
    // rather than red, which is why this used to decline every non-identity
    // format outright.
    //
    // Returning it puts the obligation on the caller and the compiler enforces
    // it: a rail that can fold this into its image view's component mapping
    // does so, and one that cannot must decline by name. Deriving it at a call
    // site instead is not available — the plan is a property of the *Metal*
    // format, and a rail holding only a `TexelLayout` or a `VkFormat` has
    // already lost the distinction between `A8Unorm` and `R8Unorm`.
    Ok((layout, srgb_decline(&f, mtl), f.components))
}

/// The Vulkan format for a guest [`TexelLayout`].
///
/// The single crossing from the decode vocabulary to the host one, applied
/// where the engine creates a sampled image. Linear by construction: a layout
/// carries no transfer function, so a rail that reaches a sampled image through
/// here has already recorded whatever [`sampled_pixels`] handed back.
pub fn vk_texel_layout(layout: TexelLayout) -> vk::Format {
    match layout {
        TexelLayout::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TexelLayout::Bgra8 => vk::Format::B8G8R8A8_UNORM,
        TexelLayout::R8 => vk::Format::R8_UNORM,
        TexelLayout::Rg8 => vk::Format::R8G8_UNORM,
        TexelLayout::R16Float => vk::Format::R16_SFLOAT,
        TexelLayout::R32Float => vk::Format::R32_SFLOAT,
        TexelLayout::R16Unorm => vk::Format::R16_UNORM,
        TexelLayout::Rg16Unorm => vk::Format::R16G16_UNORM,
        TexelLayout::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        TexelLayout::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TexelLayout::Rg16Float => vk::Format::R16G16_SFLOAT,
        TexelLayout::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        TexelLayout::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        TexelLayout::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        // The BC families. Each Metal spelling and its Vulkan counterpart are
        // the same block layout with the same bytes in the same order, so the
        // guest's payload is uploaded verbatim — which is why these need no
        // conversion arm anywhere and are admitted as one family.
        TexelLayout::Bc1Rgba => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TexelLayout::Bc2Rgba => vk::Format::BC2_UNORM_BLOCK,
        TexelLayout::Bc3Rgba => vk::Format::BC3_UNORM_BLOCK,
        TexelLayout::Bc4RUnorm => vk::Format::BC4_UNORM_BLOCK,
        TexelLayout::Bc4RSnorm => vk::Format::BC4_SNORM_BLOCK,
        TexelLayout::Bc5RgUnorm => vk::Format::BC5_UNORM_BLOCK,
        TexelLayout::Bc5RgSnorm => vk::Format::BC5_SNORM_BLOCK,
        TexelLayout::Bc6hRgbFloat => vk::Format::BC6H_SFLOAT_BLOCK,
        TexelLayout::Bc6hRgbUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        TexelLayout::Bc7Rgba => vk::Format::BC7_UNORM_BLOCK,
    }
}

/// The sRGB spelling of a guest [`TexelLayout`], for the layouts that have one.
///
/// The counterpart of [`vk_texel_layout`] for an image whose stored values are
/// sRGB-encoded, so the hardware decodes on sample. `None` for every layout that
/// cannot hold an sRGB image — see [`TexelLayout::has_srgb_encoding`], which
/// this agrees with by a `const` assertion below rather than by a second list.
///
/// Written as the inverse of [`storage_format`] and held to it by
/// `the_srgb_spelling_of_a_layout_stores_that_layout`: a pair that disagreed
/// would key a resident allocation on one format and bind a view of the other.
pub fn srgb_texel_layout(layout: TexelLayout) -> Option<vk::Format> {
    match layout {
        TexelLayout::Rgba8 => Some(vk::Format::R8G8B8A8_SRGB),
        TexelLayout::Bgra8 => Some(vk::Format::B8G8R8A8_SRGB),
        // The four BC families Apple gives an sRGB spelling. BC4/BC5 are
        // single- and two-channel data rather than colour and BC6H is HDR
        // float, so none of the three has one on either side of the boundary.
        TexelLayout::Bc1Rgba => Some(vk::Format::BC1_RGBA_SRGB_BLOCK),
        TexelLayout::Bc2Rgba => Some(vk::Format::BC2_SRGB_BLOCK),
        TexelLayout::Bc3Rgba => Some(vk::Format::BC3_SRGB_BLOCK),
        TexelLayout::Bc7Rgba => Some(vk::Format::BC7_SRGB_BLOCK),
        _ => None,
    }
}

/// The Vulkan format for bytes a CPU loader produced.
///
/// The one crossing for the [`SampledSourceRequest::Bytes`-shaped rails][b], and
/// the reason [`SampledByteFormat`] carries the source format: a
/// [`TexelLayout`] alone is linear by construction, so every CPU upload of an
/// sRGB guest texture used to reach the sampler through a `_UNORM` view while
/// the zero-copy gather rails — which carry a resolved format — bound the
/// `_SRGB` one. Same texture, same bytes, two colours, chosen by whichever rail
/// the cost decision took.
///
/// A source that is sRGB-encoded in a layout with no sRGB spelling is the third
/// case, and it is a real loss rather than an impossible one: a loader that
/// converts an sRGB texture into a layout outside the eight-bit colour orders
/// has moved the values out of the encoding's domain. That is reported on the
/// census — the site the census's own doc said could not exist until the
/// qualifier reached this far — and the linear spelling is bound, because the
/// bytes are what they are.
///
/// [b]: crate::runtime::draw::vulkan
pub fn vk_sampled_bytes(format: SampledByteFormat) -> vk::Format {
    let linear = vk_texel_layout(format.layout());
    let Some(mtl) = format.srgb_source() else {
        return linear;
    };
    match srgb_texel_layout(format.layout()) {
        Some(srgb) => srgb,
        None => {
            crate::runtime::census::srgb_census::note_downgrade(
                crate::runtime::census::srgb_census::site::SAMPLED_BYTE_UPLOAD,
                mtl,
            );
            linear
        }
    }
}

/// The [`TexelLayout`] a Vulkan format is, or `None` for a format that is not
/// one of them.
///
/// The inverse of [`vk_texel_layout`], for the engine, which holds a resolved
/// `vk::Format` for an attachment and needs the layout to ask
/// [`crate::contract::pixel_format`] how to write a texel of it. Written as a
/// search of `TexelLayout::ALL` rather than as a second `match`, so it cannot
/// disagree with the forward map and a new layout is covered the moment it is
/// added to `ALL`.
pub fn texel_layout_of(format: vk::Format) -> Option<TexelLayout> {
    // A layout describes stored bytes, and the transfer function does not change
    // them, so the fold is [`storage_format`]'s and is not spelled twice.
    let format = storage_format(format);
    TexelLayout::ALL
        .iter()
        .copied()
        .find(|&l| vk_texel_layout(l) == format)
}

/// The format an image is *allocated* in, for a requested view format.
///
/// A Metal texture view over an `IOSurface` is a second interpretation of one
/// allocation, never a second allocation. `BGRA8Unorm` and `BGRA8Unorm_sRGB`
/// name the same stored bytes and differ only in the fixed-function conversion
/// applied on render writes and sampled reads. Vulkan expresses exactly that
/// with one `VkImage` created `MUTABLE_FORMAT` and one `VkImageView` per
/// interpretation, so the allocation is keyed on this format and the transfer
/// function rides on the view.
///
/// **Folding here is what keeps one surface to one resident.** Keying an
/// allocation on the view format instead forks the resident the moment a guest
/// binds one surface through both spellings — which the guest does — and the two
/// images then alternate frame to frame, each holding half the content. That is
/// a content defect, not a colour one, and it is why this fold is not optional.
pub fn storage_format(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        // The four BC families with an sRGB spelling. Same rule one storage
        // shape over: a compressed image's blocks are identical bytes under
        // either qualifier, so both spellings must resolve to one allocation
        // and differ only in the view. `the_srgb_spelling_of_a_layout_stores_
        // that_layout` is what holds this to `srgb_texel_layout`.
        vk::Format::BC1_RGBA_SRGB_BLOCK => vk::Format::BC1_RGBA_UNORM_BLOCK,
        vk::Format::BC2_SRGB_BLOCK => vk::Format::BC2_UNORM_BLOCK,
        vk::Format::BC3_SRGB_BLOCK => vk::Format::BC3_UNORM_BLOCK,
        vk::Format::BC7_SRGB_BLOCK => vk::Format::BC7_UNORM_BLOCK,
        other => other,
    }
}

/// The two formats one resident image answers for, derived from the single
/// format the guest declared so they cannot disagree.
///
/// A resident is asked its format by two kinds of caller and they want two
/// different answers:
///
/// * **the allocation** — what `vkCreateImage` is given, what keys reuse of a
///   live slot, and what buckets the image in the recycle pool. Two declarations
///   that differ only in transfer function are one `MTLTexture` seen through two
///   `newTextureViewWithPixelFormat:` views, so they must resolve to one image.
///   That is [`storage_format`]'s rule, and this is where it is applied.
/// * **the declaration** — what a render pass attaches, and the stronger of the
///   two answers a sampled bind can be given, because it carries the transfer
///   function Vulkan applies on write and on read.
///
/// Both were spelled `color_format`, one `vk::Format` doing both jobs, and the
/// two `registry_ensure*` arms picked differently: the primary one keyed the
/// allocation on the declaration, which forks one surface into two images the
/// moment the guest binds it through both spellings; the secondary one keyed
/// reuse on the allocation while registering, creating and recycling under the
/// declaration, so an sRGB resident there was retired on every ensure and its
/// recycled image went into a bucket nothing takes from. Carrying the pair as
/// one value is what makes those two mistakes unspellable: neither answer can be
/// reached without naming which one it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentFormat(vk::Format);

impl ResidentFormat {
    /// The resident behind a guest declaration of `declared`.
    pub fn of(declared: vk::Format) -> Self {
        Self(declared)
    }

    /// What the guest declared: the render pass's attachment format, and the
    /// interpretation a sampled view of this resident decodes through.
    pub fn declared(self) -> vk::Format {
        self.0
    }

    /// The allocation family. Keys `vkCreateImage`, live-slot reuse and the
    /// recycle bucket — never the view.
    pub fn allocation(self) -> vk::Format {
        storage_format(self.0)
    }

    /// Whether the declaration adds a transfer function the allocation does not
    /// carry, so the attachment needs a view of its own over the same image.
    pub fn needs_own_view(self) -> bool {
        self.declared() != self.allocation()
    }
}

/// The format a **sampled** view of a resident must be created with, given the
/// format the bind asked for and the format the resident itself holds.
///
/// # Why the bind's own answer is not enough
///
/// A sampled bind's format is resolved from a [`TexelLayout`] — see
/// `engine::SampledResource::format` — and a layout names stored bytes and so
/// carries no transfer function at all. That is the right vocabulary for the
/// rails that upload guest bytes, because there the layout *is* everything the
/// device knows. It is not enough for the one source that has a second,
/// stronger declaration available: a `SampledSource::Target`, whose resident was
/// created by [`color_attachment`] from the guest's own `MTLPixelFormat` and
/// therefore does carry the transfer function.
///
/// Without this fold a resident the guest declared `BGRA8Unorm_sRGB` is written
/// through a `B8G8R8A8_SRGB` attachment — Vulkan encodes linear to sRGB, as
/// Metal does — and then sampled through a `B8G8R8A8_UNORM` view, which decodes
/// nothing. The still-encoded value is composited and encoded a second time by
/// the next attachment write, so the frame carries **exactly one sRGB encode too
/// many**. That is a colour defect with no counter behind it: every rail
/// succeeds, nothing declines, and the picture is washed out in the direction
/// `1.055 x^(1/2.4) - 0.055` describes.
///
/// # Why it can only ever add the transfer function
///
/// Two gates, and both are load-bearing.
///
/// [`stored_bytes_agree`] means it fires only where the two spellings differ in
/// nothing but the transfer function. A bind whose channel order or texel width
/// differs from the resident's is left exactly as it asked — that disagreement
/// is a real one and this is not the place to resolve it.
///
/// **The bind must also have had nothing to say.** Only a `requested` that is
/// already its own [`storage_format`] — a spelling with no transfer function on
/// it — is one the resident may answer for. A bind naming `B8G8R8A8_SRGB` over a
/// resident written through its linear view has stated an interpretation, and
/// Metal's contract is that a texture view's pixel format *is* the
/// interpretation for that bind; answering with the allocation's own spelling
/// drops the decode the bind asked for. Without this gate the function does not
/// add a transfer function, it replaces one side's with the other's, and it goes
/// wrong in whichever direction the resident happens to hold — which is how
/// `resident_sample_uses_the_bindings_compatible_format_view` caught it.
///
/// What is left unresolved, and said rather than hidden: a bind spelled through
/// a `TexelLayout` is linear because that vocabulary has no other spelling, so
/// this cannot tell it from a guest that genuinely asked for a linear view of an
/// sRGB surface. The resident wins there, which is right for every rail that
/// reaches here through a layout and would be wrong for a rail that could say
/// linear and meant it. Closing that needs the sampled rails to carry the
/// guest's `MTLPixelFormat` rather than a byte layout; it is not closable here.
pub fn sample_view_format(requested: vk::Format, resident: vk::Format) -> vk::Format {
    if requested == storage_format(requested) && stored_bytes_agree(requested, resident) {
        resident
    } else {
        requested
    }
}

/// Whether two Vulkan formats describe the same stored bytes for the same
/// texel, so a transfer that converts nothing may move one into the other.
///
/// This is the question a `vkCmdCopyImageToBuffer` out of a render target into
/// guest pages actually asks, and it is not format equality. The two sides
/// reaching that copy answer two different questions by design: an attachment
/// carries the guest's transfer function, because [`color_attachment`] keeps it
/// so Vulkan performs the fixed-function linear-to-sRGB encode on write, while a
/// guest destination is spelled as a [`TexelLayout`] via [`vk_texel_layout`] and
/// has no transfer function to carry. A guest render target declared
/// `BGRA8Unorm_sRGB` therefore meets itself as `B8G8R8A8_SRGB` against
/// `B8G8R8A8_UNORM`, forever, and equality reads that as a disagreement.
///
/// Vulkan is explicit that it is not one: buffer/image copies perform no format
/// conversion, so what crosses is the stored texel, and [`storage_format`] is
/// this module's existing fold onto it. Everything a byte-level comparison must
/// still separate survives that fold — channel order (`R8G8B8A8` against
/// `B8G8R8A8`) and texel width (eight-bit against half-float) both differ in the
/// storage format, not only in the view.
pub fn stored_bytes_agree(held: vk::Format, want: vk::Format) -> bool {
    storage_format(held) == storage_format(want)
}

/// Whether a Vulkan colour format stores its first and third channels in BGRA
/// order. The transfer function is deliberately irrelevant: UNORM and sRGB
/// views interpret the same four stored bytes.
pub fn has_bgra_order(format: vk::Format) -> bool {
    matches!(texel_layout_of(format), Some(TexelLayout::Bgra8))
}

/// Every Vulkan format a colour attachment may take, and the decline for a
/// format the rail does not render to.
///
/// The result is the resolved [`vk::Format`] rather than an engine enum, so an
/// sRGB target reaches an sRGB attachment and gets Vulkan's fixed-function
/// linear-to-sRGB conversion on writes.
///
/// The narrowing is deliberate and stays. Metal renders to far more formats
/// than this device carries; admitting one the rest of the pass machinery has
/// never carried would trade a named decline for a wrong picture.
///
/// **Which formats those are is the contract's answer, not this function's.**
/// [`pixel_format::render_target_bpp`] says in its own doc that "the match arms
/// *are* the renderable set", and this used to hold a second list — of Vulkan
/// formats rather than Metal ones, so nothing could compare them. They had
/// already drifted: the contract admitted `RGBA16_FLOAT`, which is what lets a
/// half-float *primary* attachment be created at the format the guest declared,
/// while this refused it. One guest format was therefore renderable as slot 0
/// and declined as a secondary MRT slot, which is not a narrowing anybody chose.
///
/// Asking the contract makes the two arms one answer, and makes adding a
/// renderable format a single edit there rather than a pair of edits that a
/// commit can half-land.
pub fn color_attachment(
    mtl: u16,
) -> Result<(vk::Format, Option<TranslateReason>), TranslateReason> {
    let f = translate(mtl)?;
    if pixel_format::render_target_bpp(mtl).is_none() {
        return Err(TranslateReason::NoColorAttachmentFormat(mtl));
    }
    Ok((f.vk, None))
}

/// The engine's storage-image format for a contract [`StorageImageSelector`].
///
/// The selector is the compute rail's own narrowing of `MTLPixelFormat`, so
/// this is a vocabulary-to-vocabulary step rather than a Metal decision — but
/// it lives here for the same reason everything else does: it was previously
/// spelled in `runtime/compute_exec/mod.rs`, where nothing could see that the two
/// enums had to stay in step.
///
/// It is **total**, and that is the point. It used to take the selector's `u32`
/// ordinal and match it with thirteen `s if s == S::X as u32` guard arms, which
/// the compiler cannot check for coverage — so a new selector variant compiled
/// fine here and declined at run time as a drift between two vocabularies that
/// had not actually drifted. Taking the enum makes the arms exhaustive and the
/// decline unnecessary: every selector the contract can produce has an engine
/// format, and a new one cannot be added without this answering for it.
pub fn storage_image_from_selector(selector: StorageImageSelector) -> StorageImageFormat {
    use crate::contract::pixel_format::StorageImageSelector as S;
    match selector {
        S::Rgba8Uint => StorageImageFormat::Rgba8Uint,
        S::Rgba8Sint => StorageImageFormat::Rgba8Sint,
        S::Rgba16Uint => StorageImageFormat::Rgba16Uint,
        S::Rgba16Float => StorageImageFormat::Rgba16Float,
        S::Rgba32Float => StorageImageFormat::Rgba32Float,
        S::Rgba8Unorm => StorageImageFormat::Rgba8Unorm,
        S::Bgra8Unorm => StorageImageFormat::Bgra8Unorm,
        S::R16Float => StorageImageFormat::R16Float,
        S::Rg16Float => StorageImageFormat::Rg16Float,
        S::R8Unorm => StorageImageFormat::R8Unorm,
        S::Rg8Unorm => StorageImageFormat::Rg8Unorm,
        S::Rgba32Uint => StorageImageFormat::Rgba32Uint,
        S::R32Uint => StorageImageFormat::R32Uint,
    }
}

/// The engine's storage-image format for a Metal pixel format.
///
/// Used by the compute rails for both storage bindings and sampled textures
/// staged through the storage selector. The four single-channel-wide formats
/// below never had a storage selector — the contract's selector enum has no
/// ordinal for them — so they are answered directly rather than being declined
/// by a narrowing they were never in.
///
/// # Why this rail keeps an enum where the others took a `VkFormat`
///
/// The colour-attachment and sampled rails resolve to a real `VkFormat` so an
/// sRGB format is expressible on them. This one does not, and the reason is not
/// inertia:
///
/// * **No sRGB format reaches it.** `pixel_format::storage_selector` has no
///   sRGB arm, so an sRGB format declines here with
///   [`TranslateReason::NoStorageImageFormat`] rather than downgrading — which
///   is why `srgb_census` names six rails and none of them is this one. Widening
///   the vocabulary would therefore make no colour space newly reachable.
/// * **The shader side cannot name one either.** A storage image's view format
///   must be class-compatible with the format the SPIR-V module declares, and
///   the SPIR-V image-format operand has no sRGB member at all
///   (`runtime::spirv_bind::ImageFormat`). Vulkan likewise does not apply the
///   transfer function on an image store.
/// * **Its consumer reasons about the enum by name.** The compute path picks a
///   view format by comparing the guest surface's format class against the
///   shader's declared one; expressed over `VkFormat` that reasoning would have
///   to be spelled in `runtime/`, which is exactly the boundary
///   `translate::gate` exists to keep closed.
///
/// A test below pins the first point, so if a future selector does admit an
/// sRGB format this comment stops being true loudly rather than quietly.
pub fn storage_image(mtl: u16) -> Result<StorageImageFormat, TranslateReason> {
    use crate::contract::pixel_format as pf;
    // Validate the format against the one pixel table first, so an entirely
    // unknown value declines as `unknown_pixel_format` rather than as a missing
    // storage layout — those are different bugs and want different slugs.
    translate(mtl)?;
    match mtl {
        pf::MTL_FORMAT_R32_UINT => return Ok(StorageImageFormat::R32Uint),
        pf::MTL_FORMAT_R32_SINT => return Ok(StorageImageFormat::R32Sint),
        pf::MTL_FORMAT_R32_FLOAT => return Ok(StorageImageFormat::R32Float),
        pf::MTL_FORMAT_RGB9E5_FLOAT => return Ok(StorageImageFormat::Rgb9e5Ufloat),
        _ => {}
    }
    let selector = pf::storage_selector(mtl).ok_or(TranslateReason::NoStorageImageFormat(mtl))?;
    Ok(storage_image_from_selector(selector))
}

/// The compute path's admission for a **sampled** image bind.
///
/// [`storage_image`] answers the storage question, and the compute rail used to
/// ask it for both roles — `mtl_to_engine_sampled` was a one-line wrapper over
/// it. That is why macOS 14 and macOS 15 each lost a whole
/// `DispatchThreadgroups` a boot to `sampled_format_unsupported` on
/// `MTLPixelFormatR16Unorm`: the ten-bit biplanar video luma plane is
/// sampleable everywhere and is not a storage format, so the storage table
/// correctly refused a question it was never being asked.
///
/// The two questions are genuinely different and Vulkan says so. `R16_UNORM` is
/// mandatory for `SAMPLED_IMAGE` with `SAMPLED_IMAGE_FILTER_LINEAR` and carries
/// no mandatory `STORAGE_IMAGE` support; `E5B9G9R9_UFLOAT_PACK32` has no storage
/// support at all. So this is a superset of [`storage_image`] rather than a copy
/// of it, and the members it adds are exactly the ones marked sampled-only on
/// [`StorageImageFormat`].
///
/// The graphics rail asks [`sampled_pixels`] instead, which answers a
/// [`TexelLayout`] and is wider still. The two are not merged because the
/// compute request carries a `StorageImageFormat` — see that type's doc for the
/// end state that would let them be.
///
/// # The two rails are held to each other by a test
///
/// Admitting `R16_UNORM` alone left the *chroma* half of the same biplanar video
/// texture refused, so the dispatch a shader makes of both planes was still lost
/// — the refusal moved to the other binding. That is the failure mode of fixing
/// a divergence one format at a time, so it is now a relation rather than a list:
/// `a_texture_the_graphics_rail_samples_is_not_refused_by_the_compute_one` sweeps
/// every `u16` and requires everything [`sampled_pixels`] admits to be admitted
/// here, against a named exception set that the test states the reason for.
///
/// The converse does not hold and must not: this rail carries the integer and
/// packed formats a compute shader reads and [`sampled_pixels`] has no
/// [`TexelLayout`] for, because that one answers a CPU-upload byte order.
pub fn sampled_image(mtl: u16) -> Result<StorageImageFormat, TranslateReason> {
    use crate::contract::pixel_format as pf;
    // Sampled-only members first, then everything a storage image may be. The
    // `translate` call keeps an entirely unknown value declining as
    // `unknown_pixel_format` rather than as a missing layout, exactly as
    // `storage_image` does for the same reason.
    let sampled_only = match mtl {
        pf::MTL_FORMAT_R16_UNORM => StorageImageFormat::R16Unorm,
        pf::MTL_FORMAT_RG16_UNORM => StorageImageFormat::Rg16Unorm,
        pf::MTL_FORMAT_RGBA16_UNORM => StorageImageFormat::Rgba16Unorm,
        pf::MTL_FORMAT_RGB10A2_UNORM => StorageImageFormat::Rgb10a2Unorm,
        pf::MTL_FORMAT_BGR10A2_UNORM => StorageImageFormat::Bgr10a2Unorm,
        pf::MTL_FORMAT_RG11B10_FLOAT => StorageImageFormat::Rg11b10Float,
        _ => return storage_image(mtl),
    };
    translate(mtl)?;
    Ok(sampled_only)
}

/// The guest's scanout byte order, in Vulkan terms.
///
/// The compositor's framebuffers are `MTLPixelFormatBGRA8Unorm`, so a resident
/// target and a swapchain image both use this format to keep
/// the present path free of a channel swap. Named once because it is one
/// decision — spelled at each site it would drift, and a single wrong spelling
/// shows up as red-and-blue-swapped output rather than a failure.
///
/// A test holds it equal to what the pixel table answers for
/// `MTL_FORMAT_BGRA8_UNORM`, so it cannot become a second opinion.
pub const SCANOUT_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// The engine's neutral resident colour format, used where content is not
/// destined straight for scanout and the channel order does not matter.
pub const RESIDENT_RGBA_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The transient depth attachment format. Depth-only passes use this; a pass
/// that also needs stencil negotiates a combined format against the device,
/// because which combined format exists is a capability question.
pub const TRANSIENT_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Resident colour format for a target, by whether its bytes must already be in
/// guest scanout order.
pub fn resident_color(bgra: bool) -> vk::Format {
    if bgra {
        SCANOUT_FORMAT
    } else {
        RESIDENT_RGBA_FORMAT
    }
}

/// Bytes occupied by one texel of a Vulkan format, for the formats this table
/// can produce.
///
/// The inverse view of [`PixelFormat::bytes_per_texel`], needed once the engine
/// stores a resolved `VkFormat` rather than a byte-layout enum that carried its
/// own size. `None` for anything outside the set — including block-compressed
/// and multi-planar formats, whose footprint is not one number per texel — so a
/// caller declines by name instead of computing a wrong buffer size.
///
/// An sRGB format has the footprint of its linear sibling, which is what makes
/// flipping a rail to sRGB a pure colour-space change with no allocation
/// consequences.
/// Every [`TexelLayout`] answers from the contract's own width, so a layout
/// added to [`TexelLayout::ALL`] is covered here the moment it exists. This was
/// a hand-kept second copy of those widths, and it was missing
/// `R16G16B16A16_UNORM` for as long as that layout had existed — which cost
/// macOS 26 a hundred and eight draws a boot, because a width this table did not
/// know is indistinguishable from a block-compressed one and declines by the
/// same name. Same argument as [`texel_layout_of`] being a search rather than a
/// second `match`.
/// The storage **block** grid of a Vulkan format, for the formats this table can
/// produce.
///
/// [`bytes_per_texel`] with the grid stated, and derived from the same
/// [`texel_layout_of`] search so the two cannot disagree. A caller sizing a
/// linear buffer for an image must ask this rather than `bytes_per_texel`:
/// multiplying a block byte count by width and height over-counts a compressed
/// image by sixteen, which is a refusal against the guest's own correctly-sized
/// buffer rather than a wrong image.
///
/// sRGB spellings fold through [`storage_format`] onto the allocation they share
/// with their linear sibling. That fold is what covers the four `BC*_SRGB_BLOCK`
/// formats, which a sampled bind of an sRGB compressed texture is created as.
pub fn vk_block_geometry(format: vk::Format) -> Option<pixel_format::BlockGeometry> {
    if let Some(layout) = texel_layout_of(storage_format(format)) {
        return Some(layout.block());
    }
    Some(pixel_format::BlockGeometry {
        width: 1,
        height: 1,
        bytes: bytes_per_texel(format)?,
    })
}

pub fn bytes_per_texel(format: vk::Format) -> Option<u32> {
    if let Some(layout) = texel_layout_of(format) {
        return Some(layout.bytes_per_texel());
    }
    // What remains is the formats that are deliberately not `TexelLayout`s:
    // depth/stencil, the packed shared-exponent float, and the integer and sRGB
    // spellings of the colour orders. None is a guest linear texel layout, so
    // none has a contract width to derive from.
    Some(match format {
        vk::Format::S8_UINT => 1,
        vk::Format::D16_UNORM => 2,
        vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::E5B9G9R9_UFLOAT_PACK32
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT => 4,
        vk::Format::R16G16B16A16_UINT | vk::Format::D32_SFLOAT_S8_UINT => 8,
        vk::Format::R32G32B32A32_UINT | vk::Format::R32G32B32A32_SFLOAT => 16,
        _ => return None,
    })
}

/// The Vulkan spelling of an engine storage/compute image format.
pub fn vk_storage_image(format: StorageImageFormat) -> vk::Format {
    match format {
        StorageImageFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        StorageImageFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        StorageImageFormat::R16Float => vk::Format::R16_SFLOAT,
        StorageImageFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        StorageImageFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        StorageImageFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        StorageImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        StorageImageFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        StorageImageFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        StorageImageFormat::R8Unorm => vk::Format::R8_UNORM,
        StorageImageFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        StorageImageFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        StorageImageFormat::R32Uint => vk::Format::R32_UINT,
        StorageImageFormat::R32Sint => vk::Format::R32_SINT,
        StorageImageFormat::R32Float => vk::Format::R32_SFLOAT,
        StorageImageFormat::Rgb9e5Ufloat => vk::Format::E5B9G9R9_UFLOAT_PACK32,
        StorageImageFormat::R16Unorm => vk::Format::R16_UNORM,
        StorageImageFormat::Rg16Unorm => vk::Format::R16G16_UNORM,
        StorageImageFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        StorageImageFormat::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        StorageImageFormat::Bgr10a2Unorm => vk::Format::A2R10G10B10_UNORM_PACK32,
        StorageImageFormat::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
    }
}

/// The decline a rail owes its caller when it binds the linear sibling of an
/// sRGB Metal format.
///
/// On the layout rails ([`sampled_pixels`], [`storage_image`]) the loss is
/// structural — a byte layout has no transfer function to carry. On
/// [`color_attachment`], which now resolves to a real `VkFormat`, it is a
/// deliberate hold: the crate ignores the transfer function consistently, so
/// flipping one rail on its own would break that symmetry.
fn srgb_decline(f: &PixelFormat, mtl: u16) -> Option<TranslateReason> {
    f.is_srgb().then_some(TranslateReason::SrgbDowngraded(mtl))
}

/// A decoded swizzle plan as the `VkImageView` component mapping that performs
/// it in hardware.
///
/// The plan passed in is **already folded**: the caller composes the decoded
/// type-8 view swizzle over the format's own channel remap with
/// [`crate::contract::pixel_format::SwizzlePlan::after`], because a
/// `VkComponentMapping` can express one plan and a bind may need both. This
/// function does no composing of its own and must not start — it would then be
/// a second place the fold happens, and the two would disagree the first time
/// only one of them was updated.
///
/// It used to be the view swizzle alone, which was safe only because
/// [`sampled_pixels`] refused every format with a non-identity plan. It no
/// longer refuses them: it returns the plan, and `A8Unorm` — whose byte rides
/// in `R8_UNORM` — is sampled rather than sent to the CPU rung.
///
/// This is what makes a swizzled view cost nothing: Vulkan applies the mapping
/// at sample time, so the texels never have to be rewritten on the CPU (which
/// would force the whole texture onto the CPU upload path and cost the
/// zero-copy property for it).
pub fn vk_component_mapping(plan: &SwizzlePlan) -> vk::ComponentMapping {
    fn one(source: SwizzleSource) -> vk::ComponentSwizzle {
        match source {
            SwizzleSource::Zero => vk::ComponentSwizzle::ZERO,
            SwizzleSource::One => vk::ComponentSwizzle::ONE,
            SwizzleSource::R => vk::ComponentSwizzle::R,
            SwizzleSource::G => vk::ComponentSwizzle::G,
            SwizzleSource::B => vk::ComponentSwizzle::B,
            SwizzleSource::A => vk::ComponentSwizzle::A,
        }
    }
    vk::ComponentMapping {
        r: one(plan.source[COMPONENT_R]),
        g: one(plan.source[COMPONENT_G]),
        b: one(plan.source[COMPONENT_B]),
        a: one(plan.source[COMPONENT_A]),
    }
}

/// Whether a Metal format's channels sit identically on its Vulkan format.
///
/// The component plan is a property of the **Metal** format, not of the Vulkan
/// one it resolves to: `A8Unorm` and `R8Unorm` both land on `R8_UNORM`, and
/// only the first needs its byte moved back into alpha. A rail that has already
/// reduced a format to a host format or a byte layout can therefore no longer
/// derive the plan, which is exactly why [`sampled_pixels`] declines a
/// non-identity format instead of admitting one it could not describe.
///
/// The sampled rail relies on that: it takes the plan from [`sampled_pixels`]
/// and folds it under the decoded type-8 swizzle (see [`vk_component_mapping`]),
/// which it can only do because the plan travels with the layout instead of
/// being re-derived downstream. This predicate is now a *reader* of the same
/// fact rather than a gate on it, and a test holds the two in agreement.
pub fn has_identity_components(mtl: u16) -> bool {
    translate(mtl)
        .map(|f| f.components == IDENTITY)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Decline;
    use pixel_format as p;

    /// [`vk_texel_layout`] names stored bytes and so never carries a transfer
    /// function.
    ///
    /// The property matters where a format built by that function is asked a
    /// channel-order question: `gva_resident_format`'s output reaches
    /// `GvaTargetKey` on two sides of the GVA store witness, and both used to
    /// spell the question as `== SCANOUT_FORMAT`. That spelling is only right
    /// while this holds, and it is wrong the moment a caller passes a format
    /// from anywhere else — which is exactly what happened one rail over, in
    /// `engine::ResidentReadSnapshot::bgra`, where the format came from the
    /// attachment and did carry one. Both sites ask [`has_bgra_order`] now; this
    /// says the switch changed no answer.
    #[test]
    fn a_stored_texel_layout_never_names_a_transfer_function() {
        for &layout in TexelLayout::ALL {
            let f = vk_texel_layout(layout);
            assert_eq!(storage_format(f), f, "{layout:?}");
            assert_eq!(has_bgra_order(f), f == SCANOUT_FORMAT, "{layout:?}");
        }
    }

    /// A sampled bind can only ever spell a stored-byte format, so every
    /// resident the guest declared with an sRGB format would be sampled without
    /// its decode if the bind's own answer were taken. `sample_view_format`
    /// restores it, for both channel orders and both spellings of the bind.
    #[test]
    fn a_sampled_view_takes_the_transfer_function_from_the_resident() {
        for (linear, srgb) in [
            (SCANOUT_FORMAT, vk::Format::B8G8R8A8_SRGB),
            (RESIDENT_RGBA_FORMAT, vk::Format::R8G8B8A8_SRGB),
        ] {
            // The defect: the bind names stored bytes, the resident carries the
            // qualifier, and the view has to be the resident's.
            assert_eq!(sample_view_format(linear, srgb), srgb);
            // Already agreed, either way round: nothing to restore.
            assert_eq!(sample_view_format(srgb, srgb), srgb);
            assert_eq!(sample_view_format(linear, linear), linear);
            // A bind that spelled the qualifier itself keeps it. The comment
            // beside this line already said so — "this must not become a
            // downgrade of a bind that already spelled it" — while the assertion
            // under it demanded the downgrade, and the engine obeyed the
            // assertion: a resident written through its linear attachment view
            // and sampled through its sRGB sibling was bound linear and decoded
            // nothing.
            assert_eq!(sample_view_format(srgb, linear), srgb);
        }
    }

    /// The fold may add a transfer function and nothing else. A bind whose
    /// channel order or texel width disagrees with the resident is a real
    /// disagreement and is left exactly as it asked — resolving it here would
    /// silently rewrite what the shader samples.
    #[test]
    fn a_sampled_view_never_changes_the_stored_bytes_the_bind_asked_for() {
        for &layout in TexelLayout::ALL {
            let requested = vk_texel_layout(layout);
            for &resident in TexelLayout::ALL {
                for resident in [
                    vk_texel_layout(resident),
                    // Both sRGB spellings a resident can hold, so the case that
                    // matters is exercised against every requested layout.
                    vk::Format::B8G8R8A8_SRGB,
                    vk::Format::R8G8B8A8_SRGB,
                ] {
                    let got = sample_view_format(requested, resident);
                    assert!(
                        stored_bytes_agree(got, requested),
                        "{requested:?} against {resident:?} became {got:?}"
                    );
                }
            }
        }
    }

    /// The copy out of a render target into guest pages converts nothing, so the
    /// only thing the two sides must agree on is the stored texel.
    ///
    /// Every guest render target declared with an sRGB format meets itself
    /// across this comparison — the attachment keeps the transfer function so
    /// Vulkan encodes on write, and the guest destination is spelled as a bare
    /// [`TexelLayout`] — so format equality here refuses a copy whose bytes are
    /// identical. On a driven macos-13 Maps leg that was 1 001 refusals out of
    /// 1 001, and the app's canvas kept the zeros its pages were allocated with.
    #[test]
    fn a_transfer_function_is_not_a_disagreement_about_stored_bytes() {
        for (view, stored) in [
            (vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM),
        ] {
            assert!(
                stored_bytes_agree(view, stored),
                "{view:?} and {stored:?} are one allocation seen two ways"
            );
            assert!(stored_bytes_agree(stored, view), "the rule is symmetric");
            assert!(stored_bytes_agree(view, view));
        }
    }

    /// What the comparison is *for* survives the fold. Channel order and texel
    /// width are storage facts, not view facts, so folding the transfer function
    /// cannot admit either — a BGRA resident under an RGBA destination would need
    /// an exchange this copy cannot perform, and a half-float destination over an
    /// eight-bit resident would overlap its rows at half their true pitch.
    #[test]
    fn channel_order_and_texel_width_still_disagree() {
        for (a, b) in [
            (vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB),
            (vk::Format::B8G8R8A8_UNORM, vk::Format::R16G16B16A16_SFLOAT),
            (vk::Format::B8G8R8A8_SRGB, vk::Format::R16G16B16A16_SFLOAT),
        ] {
            assert!(
                !stored_bytes_agree(a, b),
                "{a:?} and {b:?} do not store the same bytes for one texel"
            );
        }
    }

    /// A guest texture the graphics rail will sample is one the compute rail
    /// will sample, for every `MTLPixelFormat` value there is.
    ///
    /// This is the relation two separate bugs were instances of. The same guest
    /// texture, sampleable in a draw and refused in a dispatch, costs the whole
    /// `DispatchThreadgroups` — and finding those one format at a time does not
    /// converge: admitting `R16_UNORM` left the chroma half of the very same
    /// biplanar video texture refused, so the loss simply moved to the shader's
    /// other binding. Sweeping every `u16` is what makes the next one a failure
    /// here rather than a lost frame on a rail nobody booted.
    ///
    /// The exceptions are listed rather than tolerated, because each is a real
    /// decision this rail cannot yet express:
    ///
    /// - `A8Unorm` needs its channel plan. [`sampled_pixels`] hands back a
    ///   [`SwizzlePlan`] that puts the single byte in alpha; a
    ///   [`StorageImageFormat`] carries no component mapping, so admitting it
    ///   here would sample the byte as **red**. A wrong sample is worse than a
    ///   named refusal, so it stays refused until the compute request can carry
    ///   a plan.
    /// - The two `*_SRGB` orders would have to bind their linear sibling, which
    ///   is the [`TranslateReason::SrgbDowngraded`] loss [`srgb_decline`]
    ///   documents. This rail's `Result` has nowhere to record it, and
    ///   [`storage_image`] refuses sRGB for the same reason with its own test
    ///   pinning that. Admitting it silently here would break the symmetry the
    ///   crate relies on, so it waits for the rail to gain a warning channel.
    ///
    /// - The **BC block-compressed families** cannot cross this rail at all,
    ///   and that is structural rather than pending a channel. A
    ///   [`StorageImageFormat`] is what a compute *storage* binding is created
    ///   as, and Vulkan has no block-compressed storage-image format — a shader
    ///   cannot write a block. The compute rail routes its sampled textures
    ///   through that same selector, so a compressed texture sampled inside a
    ///   dispatch is refused by name. Giving it a rail of its own means a
    ///   compute sampled path that does not go through the storage selector,
    ///   which is a change to that rail and not to this table.
    ///
    ///   Measured on the workload that brought the family in: Asphalt 8 samples
    ///   its BC3 textures from **fragment** shaders only, so this refusal cost
    ///   nothing there. A guest that samples one in a dispatch loses that
    ///   dispatch's texture and says so.
    ///
    /// The converse is deliberately not asserted: this rail carries the integer
    /// and packed formats a compute shader reads and [`sampled_pixels`] has no
    /// [`TexelLayout`] for, because that one answers a CPU-upload byte order and
    /// not a sampling capability.
    #[test]
    fn a_texture_the_graphics_rail_samples_is_not_refused_by_the_compute_one() {
        const EXCEPTIONS: &[(u16, &str)] = &[
            (p::MTL_FORMAT_A8_UNORM, "needs a component mapping"),
            (p::MTL_FORMAT_RGBA8_UNORM_SRGB, "would downgrade unrecorded"),
            (p::MTL_FORMAT_BGRA8_UNORM_SRGB, "would downgrade unrecorded"),
            (p::MTL_FORMAT_BC1_RGBA, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC1_RGBA_SRGB, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC2_RGBA, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC2_RGBA_SRGB, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC3_RGBA, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC3_RGBA_SRGB, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC4_R_UNORM, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC4_R_SNORM, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC5_RG_UNORM, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC5_RG_SNORM, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC6H_RGB_FLOAT, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC6H_RGB_UFLOAT, "no block-compressed storage image exists"),
            (p::MTL_FORMAT_BC7_RGBA_UNORM, "no block-compressed storage image exists"),
            (
                p::MTL_FORMAT_BC7_RGBA_UNORM_SRGB,
                "no block-compressed storage image exists",
            ),
        ];

        let mut refused = Vec::new();
        for mtl in 0..=u16::MAX {
            if sampled_pixels(mtl).is_ok() && sampled_image(mtl).is_err() {
                refused.push(mtl);
            }
        }
        let expected: Vec<u16> = EXCEPTIONS.iter().map(|&(mtl, _)| mtl).collect();
        assert_eq!(
            refused, expected,
            "a format the graphics rail samples must be sampleable in a dispatch \
             or be one of the exceptions this test names"
        );

        // Each exception is refused for the reason claimed and not because the
        // contract does not define it — an undefined value would satisfy the
        // sweep above for the wrong reason.
        for &(mtl, why) in EXCEPTIONS {
            assert!(translate(mtl).is_ok(), "{mtl:#x} is a defined format ({why})");
        }

        // The ten-bit biplanar video planes travel together: a shader samples
        // luma and chroma from one frame, so one admitted without the other is
        // the whole dispatch lost anyway.
        assert_eq!(
            sampled_image(p::MTL_FORMAT_R16_UNORM),
            Ok(StorageImageFormat::R16Unorm)
        );
        assert_eq!(
            sampled_image(p::MTL_FORMAT_RG16_UNORM),
            Ok(StorageImageFormat::Rg16Unorm)
        );

        // Sampled-only means sampled-only: none of the members this rail adds
        // over the storage one may be reached as a storage image, because Vulkan
        // mandates none of them for `STORAGE_IMAGE`.
        for mtl in [
            p::MTL_FORMAT_R16_UNORM,
            p::MTL_FORMAT_RG16_UNORM,
            p::MTL_FORMAT_RGBA16_UNORM,
        ] {
            assert!(
                storage_image(mtl).is_err(),
                "{mtl:#x} is sampled-only and must not be admitted as a storage image"
            );
        }
    }

    /// `texel_layout_of` is `vk_texel_layout` read backwards, for every layout
    /// and for nothing else.
    ///
    /// The round trip is the whole property: the engine holds a resolved
    /// `vk::Format` for an attachment and uses this to ask the contract how to
    /// write a texel of it, so a layout that does not come back is one whose
    /// seed would be staged at the wrong width. Driven from `TexelLayout::ALL`
    /// so a new layout is covered without anyone adding a line.
    #[test]
    fn every_texel_layout_survives_the_round_trip_through_its_vulkan_format() {
        for &layout in TexelLayout::ALL {
            assert_eq!(
                texel_layout_of(vk_texel_layout(layout)),
                Some(layout),
                "{layout:?} does not come back from its own format"
            );
        }
        // A format that is not a texel layout answers `None` rather than the
        // nearest one; depth is the case the engine could plausibly present.
        assert_eq!(texel_layout_of(TRANSIENT_DEPTH_FORMAT), None);
        assert_eq!(texel_layout_of(vk::Format::UNDEFINED), None);
    }

    /// Every `MTLPixelFormat` the decode contract defines, with the Vulkan
    /// format and texel size it must produce. Written out literally rather than
    /// derived from the table under test, so a mistranslation shows up as a
    /// diff instead of agreeing with itself.
    const EXPECTED: &[(u16, vk::Format, u32, TransferFunction)] = &[
        (
            p::MTL_FORMAT_A8_UNORM,
            vk::Format::R8_UNORM,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R8_UNORM,
            vk::Format::R8_UNORM,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R8_UINT,
            vk::Format::R8_UINT,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R16_FLOAT,
            vk::Format::R16_SFLOAT,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG8_UNORM,
            vk::Format::R8G8_UNORM,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG8_UINT,
            vk::Format::R8G8_UINT,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R32_UINT,
            vk::Format::R32_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R32_SINT,
            vk::Format::R32_SINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_R32_FLOAT,
            vk::Format::R32_SFLOAT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG16_FLOAT,
            vk::Format::R16G16_SFLOAT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
            vk::Format::R8G8B8A8_SRGB,
            4,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_RGBA8_UINT,
            vk::Format::R8G8B8A8_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA8_SINT,
            vk::Format::R8G8B8A8_SINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BGRA8_UNORM,
            vk::Format::B8G8R8A8_UNORM,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            4,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_RGB10A2_UNORM,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG11B10_FLOAT,
            vk::Format::B10G11R11_UFLOAT_PACK32,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BGR10A2_UNORM,
            vk::Format::A2R10G10B10_UNORM_PACK32,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGB9E5_FLOAT,
            vk::Format::E5B9G9R9_UFLOAT_PACK32,
            4,
            TransferFunction::Linear,
        ),
        // The BC families. The width column is bytes per 4x4 **block** for
        // these, which is what `pixel_format::block_geometry` says and what the
        // sampled rail sizes rows and images from; the uncompressed rows above
        // are the same field with a 1x1 block.
        (
            p::MTL_FORMAT_BC1_RGBA,
            vk::Format::BC1_RGBA_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC1_RGBA_SRGB,
            vk::Format::BC1_RGBA_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_BC2_RGBA,
            vk::Format::BC2_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC2_RGBA_SRGB,
            vk::Format::BC2_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_BC3_RGBA,
            vk::Format::BC3_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC3_RGBA_SRGB,
            vk::Format::BC3_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_BC4_R_UNORM,
            vk::Format::BC4_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC4_R_SNORM,
            vk::Format::BC4_SNORM_BLOCK,
            p::BC_BLOCK_BYTES_8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC5_RG_UNORM,
            vk::Format::BC5_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC5_RG_SNORM,
            vk::Format::BC5_SNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC6H_RGB_FLOAT,
            vk::Format::BC6H_SFLOAT_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC6H_RGB_UFLOAT,
            vk::Format::BC6H_UFLOAT_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC7_RGBA_UNORM,
            vk::Format::BC7_UNORM_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_BC7_RGBA_UNORM_SRGB,
            vk::Format::BC7_SRGB_BLOCK,
            p::BC_BLOCK_BYTES_16,
            TransferFunction::Srgb,
        ),
        (
            p::MTL_FORMAT_R16_UNORM,
            vk::Format::R16_UNORM,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RG16_UNORM,
            vk::Format::R16G16_UNORM,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA16_UNORM,
            vk::Format::R16G16B16A16_UNORM,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA16_UINT,
            vk::Format::R16G16B16A16_UINT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA16_FLOAT,
            vk::Format::R16G16B16A16_SFLOAT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA32_UINT,
            vk::Format::R32G32B32A32_UINT,
            16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_RGBA32_FLOAT,
            vk::Format::R32G32B32A32_SFLOAT,
            16,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH16_UNORM,
            vk::Format::D16_UNORM,
            2,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH32_FLOAT,
            vk::Format::D32_SFLOAT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_STENCIL8,
            vk::Format::S8_UINT,
            1,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH24_UNORM_STENCIL8,
            vk::Format::D24_UNORM_S8_UINT,
            4,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            vk::Format::D32_SFLOAT_S8_UINT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_X32_STENCIL8,
            vk::Format::D32_SFLOAT_S8_UINT,
            8,
            TransferFunction::Linear,
        ),
        (
            p::MTL_FORMAT_X24_STENCIL8,
            vk::Format::D24_UNORM_S8_UINT,
            4,
            TransferFunction::Linear,
        ),
    ];

    /// The table is total over the contract: every defined value maps, with the
    /// expected Vulkan format and texel size.
    #[test]
    fn every_contract_pixel_format_translates() {
        for (mtl, vkf, bpt, transfer) in EXPECTED {
            let got = translate(*mtl).unwrap_or_else(|e| panic!("MTL {mtl:#x}: {e}"));
            assert_eq!(got.vk, *vkf, "MTL {mtl:#x} vk format");
            assert_eq!(got.bytes_per_texel, *bpt, "MTL {mtl:#x} texel size");
            assert_eq!(got.transfer, *transfer, "MTL {mtl:#x} transfer function");
        }
    }

    /// The texel size this module reports is the decode contract's, not a
    /// second opinion — the drift `byte_size`-beside-`vk_format` was written to
    /// prevent.
    ///
    /// Compared against the contract's **block** size rather than its
    /// bytes-per-texel. For every uncompressed format those are the same number
    /// — the block is 1x1 — and for the BC families only the block form exists,
    /// because a BC1 texel is half a byte and `bytes_per_pixel` says `None` on
    /// purpose. So this is the stronger reading of the same invariant, not a
    /// weakened one.
    #[test]
    fn texel_size_agrees_with_the_decode_contract() {
        for (mtl, _, _, _) in EXPECTED {
            assert_eq!(
                Some(translate(*mtl).unwrap().bytes_per_texel),
                pixel_format::block_geometry(*mtl).map(|block| block.bytes),
                "MTL {mtl:#x}"
            );
        }
    }

    /// The whole point of L1: an sRGB Metal format reaches an sRGB VkFormat.
    /// If this ever fails, the hardware has silently stopped applying the
    /// transfer function on write and blending is happening in the wrong
    /// colour space.
    #[test]
    fn srgb_formats_reach_an_srgb_vk_format() {
        for mtl in [
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
        ] {
            let f = translate(mtl).unwrap();
            assert!(f.is_srgb(), "MTL {mtl:#x} lost its sRGB classification");
            assert!(
                matches!(f.vk, vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB),
                "MTL {mtl:#x} mapped to non-sRGB {:?}",
                f.vk
            );
        }
    }

    /// An sRGB format's linear sibling keeps the channel order and bit layout —
    /// downgrading may cost the transfer function and nothing else, or the
    /// stored bytes stop meaning the same thing.
    #[test]
    fn the_linear_sibling_keeps_the_channel_order() {
        let rgba = translate(p::MTL_FORMAT_RGBA8_UNORM_SRGB).unwrap();
        assert_eq!(rgba.linear_vk, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            rgba.linear_vk,
            translate(p::MTL_FORMAT_RGBA8_UNORM).unwrap().vk
        );
        let bgra = translate(p::MTL_FORMAT_BGRA8_UNORM_SRGB).unwrap();
        assert_eq!(bgra.linear_vk, vk::Format::B8G8R8A8_UNORM);
        assert_eq!(
            bgra.linear_vk,
            translate(p::MTL_FORMAT_BGRA8_UNORM).unwrap().vk
        );
        assert_eq!(rgba.bytes_per_texel, 4);
        assert_eq!(bgra.bytes_per_texel, 4);
    }

    /// An undefined wire value declines by name instead of reaching a default.
    #[test]
    fn an_unknown_format_declines_by_name() {
        let err = translate(0xffff).unwrap_err();
        assert_eq!(err, TranslateReason::UnknownPixelFormat(0xffff));
        assert_eq!(err.slug(), "unknown_pixel_format");
        assert!(!is_srgb(0xffff));
        assert!(sampled_pixels(0xffff).is_err());
    }

    /// The constant-fold shortcuts stay honest against the full translation —
    /// this module and the decode contract must not hold two opinions about
    /// which formats are sRGB.
    #[test]
    fn is_srgb_tracks_the_translated_transfer_function() {
        for (mtl, _, _, transfer) in EXPECTED {
            assert_eq!(
                is_srgb(*mtl),
                *transfer == TransferFunction::Srgb,
                "MTL {mtl:#x}"
            );
            assert_eq!(
                is_srgb(*mtl),
                pixel_format::is_srgb(*mtl),
                "MTL {mtl:#x} disagrees with the decode contract"
            );
        }
    }

    /// [`sampled_pixels`] answers a bare [`TexelLayout`], which by construction
    /// has no transfer function, so it still owes its caller the decline. What
    /// changed is that the sampled *rails* no longer lose it: they pair the
    /// layout with the source format in a [`SampledByteFormat`] and
    /// [`vk_sampled_bytes`] applies it. Colour attachments keep sRGB outright.
    #[test]
    fn an_srgb_format_never_reaches_a_linear_one_silently() {
        for mtl in [
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
        ] {
            let (_, decline, _) = sampled_pixels(mtl).unwrap();
            assert_eq!(
                decline,
                Some(TranslateReason::SrgbDowngraded(mtl)),
                "sampled rail dropped sRGB with no decline"
            );

            let (format, decline) = color_attachment(mtl).unwrap();
            assert!(matches!(
                format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            ));
            assert_eq!(decline, None, "colour attachment must preserve sRGB");
        }
        // The converse: a linear format must never produce the decline, or the
        // proxy floods and stops meaning anything.
        for (mtl, _, _, transfer) in EXPECTED {
            if *transfer == TransferFunction::Srgb {
                continue;
            }
            if let Ok((_, decline, _)) = sampled_pixels(*mtl) {
                assert_eq!(decline, None, "MTL {mtl:#x}");
            }
            if let Ok((_, decline)) = color_attachment(*mtl) {
                assert_eq!(decline, None, "MTL {mtl:#x}");
            }
        }
    }

    /// The sRGB spelling of a layout is the same allocation as its linear one,
    /// and exists for exactly the layouts the contract says can hold an sRGB
    /// image.
    ///
    /// Both halves matter and neither is a restatement. The first keeps
    /// [`srgb_texel_layout`] the inverse of [`storage_format`]: a pair that
    /// disagreed would key a resident allocation on one format while binding a
    /// view of the other, which is the fork [`storage_format`]'s own doc
    /// describes. The second keeps it in step with
    /// [`TexelLayout::has_srgb_encoding`], which is what
    /// [`vk_sampled_bytes`] consults to decide whether an sRGB source is
    /// honourable or has to be reported — a layout answering `true` there with
    /// no spelling here would report a downgrade it could have avoided, and one
    /// answering `false` with a spelling here would hide a real one.
    #[test]
    fn the_srgb_spelling_of_a_layout_stores_that_layout() {
        for layout in TexelLayout::ALL.iter().copied() {
            match srgb_texel_layout(layout) {
                Some(srgb) => {
                    assert_eq!(
                        storage_format(srgb),
                        vk_texel_layout(layout),
                        "{layout:?}: the sRGB view must store the linear allocation"
                    );
                    assert!(
                        layout.has_srgb_encoding(),
                        "{layout:?}: spelled sRGB but the contract says it cannot hold one"
                    );
                }
                None => assert!(
                    !layout.has_srgb_encoding(),
                    "{layout:?}: can hold an sRGB image and has no spelling for it"
                ),
            }
        }
    }

    /// The two ends of the CPU sampled rail meet: a source the contract calls
    /// sRGB reaches an `_SRGB` Vulkan format, and a linear one does not.
    ///
    /// This is the divergence the type exists to close. The zero-copy gather
    /// rails resolve `translate(declared).vk` and decode; the CPU rung reaches
    /// [`vk_sampled_bytes`] and must land on the same colour space, or one guest
    /// texture gets two different colours and a cost threshold picks which.
    #[test]
    fn the_cpu_sampled_rail_lands_where_the_zero_copy_rail_does() {
        for (mtl, expected) in [
            (
                p::MTL_FORMAT_RGBA8_UNORM_SRGB,
                vk::Format::R8G8B8A8_SRGB,
            ),
            (
                p::MTL_FORMAT_BGRA8_UNORM_SRGB,
                vk::Format::B8G8R8A8_SRGB,
            ),
        ] {
            let (layout, _, _) = sampled_pixels(mtl).expect("both sRGB orders sample");
            let bytes = SampledByteFormat::from_source(layout, mtl);
            assert_eq!(bytes.srgb_source(), Some(mtl));
            assert_eq!(
                vk_sampled_bytes(bytes),
                expected,
                "MTL {mtl:#x}: the CPU rung must decode where the gather rail does"
            );
            // And that is the format the zero-copy rail resolves independently.
            assert_eq!(translate(mtl).unwrap().vk, expected);
        }
        // Bytes with no guest format behind them stay linear: a clear colour is
        // stated in the space the attachment decodes to, so encoding it here
        // would apply a transfer function the guest never asked for.
        assert_eq!(
            vk_sampled_bytes(SampledByteFormat::synthesised(TexelLayout::Rgba8)),
            vk::Format::R8G8B8A8_UNORM
        );
        // A linear guest source likewise.
        assert_eq!(
            vk_sampled_bytes(SampledByteFormat::from_source(
                TexelLayout::Bgra8,
                p::MTL_FORMAT_BGRA8_UNORM
            )),
            vk::Format::B8G8R8A8_UNORM
        );
    }

    /// The storage fold changes the transfer function and **never** the channel
    /// order.
    ///
    /// This is the property that makes [`storage_format`] safe to key an
    /// allocation on. A fold that swapped `B8G8R8A8_SRGB` onto an `R8G8B8A8`
    /// storage format would put every texel's red and blue in each other's
    /// bytes, which reaches the screen as a hue rotation and nothing in the
    /// engine would refuse it — an image and a view in the same compatibility
    /// class are both valid Vulkan.
    #[test]
    fn the_storage_fold_never_changes_channel_order() {
        for &format in &[
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_UNORM,
        ] {
            let storage = storage_format(format);
            assert_eq!(
                has_bgra_order(storage),
                has_bgra_order(format),
                "{format:?} changed channel order on the way to {storage:?}"
            );
            assert_eq!(
                texel_layout_of(storage),
                texel_layout_of(format),
                "{format:?} changed byte layout on the way to {storage:?}"
            );
        }
        assert_eq!(
            storage_format(vk::Format::B8G8R8A8_SRGB),
            vk::Format::B8G8R8A8_UNORM
        );
        assert_eq!(
            storage_format(vk::Format::R8G8B8A8_SRGB),
            vk::Format::R8G8B8A8_UNORM
        );
    }

    /// Every format the forward map produces is already a storage format, and
    /// the fold is idempotent.
    ///
    /// Together these say an allocation keyed on [`storage_format`] has exactly
    /// one spelling per compatibility class, which is what stops one surface
    /// from being resident twice.
    #[test]
    fn the_storage_fold_is_idempotent_and_closed_over_the_forward_map() {
        for &layout in TexelLayout::ALL {
            let format = vk_texel_layout(layout);
            assert_eq!(
                storage_format(format),
                format,
                "{layout:?} maps to {format:?}, which is not its own storage format"
            );
        }
        for &format in &[
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R16G16B16A16_SFLOAT,
        ] {
            let once = storage_format(format);
            assert_eq!(storage_format(once), once, "{format:?} folds twice");
        }
    }

    /// Every renderable declaration's allocation is a [`TexelLayout`] this
    /// device can name.
    ///
    /// `TargetIdentity::resident_format`'s doc calls itself "the answer
    /// `registry_ensure` creates the image with", and
    /// `draw::vulkan::gva_resident_format` is what has to make that true: it
    /// takes the same `color_attachment` result the image is built from, folds
    /// it here, and then asks the host about the resulting layout. That last
    /// step is only total while this holds.
    ///
    /// It did not. `gva_resident_format` used to ask `store_texel_order`, which
    /// is the *writeback* question — can these texels be byte-copied into guest
    /// pages — and answers for three formats where `render_target_bpp` admits
    /// six. `R8Unorm`, `R16Float` and `RG16Float` render targets therefore got
    /// an identity claiming `RESIDENT_RGBA_FORMAT` over an image built at their
    /// own width, and two of the three are in the guest's vocabulary on boots on
    /// record. Two independently-maintained tables, so this is the relation
    /// between them; walking every `u16` means a format added to one and not the
    /// other cannot slip past by being absent from a hand-written list.
    #[test]
    fn every_renderable_declaration_folds_onto_a_layout_this_device_names() {
        for mtl in 0..=u16::MAX {
            let Ok((attachment, _)) = color_attachment(mtl) else {
                continue;
            };
            let allocation = ResidentFormat::of(attachment).allocation();
            assert!(
                texel_layout_of(allocation).is_some(),
                "renderable {mtl:#x} allocates as {allocation:?}, which no \
                 TexelLayout names — its resident identity cannot describe it"
            );
            assert_eq!(
                bytes_per_texel(allocation),
                Some(p::render_target_bpp(mtl).expect("color_attachment admitted it")),
                "{mtl:#x}: the allocation and the contract disagree on width"
            );
        }
    }

    /// A resident's two answers are the two questions the registry asks, and
    /// only the sRGB pair may separate them.
    ///
    /// The second half is what makes the type cheap: on every format this
    /// device renders to except the two sRGB spellings, the allocation and the
    /// declaration are the same `vk::Format`, so no extra view is ever created
    /// and `needs_own_view` answers false. A change that made some third format
    /// fold would show up here as a new pair rather than as a silent extra view
    /// per resident.
    #[test]
    fn a_residents_allocation_and_declaration_part_only_on_the_transfer_function() {
        for (declared, allocation) in [
            (vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM),
        ] {
            let f = ResidentFormat::of(declared);
            assert_eq!(f.declared(), declared);
            assert_eq!(f.allocation(), allocation);
            assert!(f.needs_own_view(), "{declared:?}");
            // The two spellings of one surface reach one allocation, which is
            // the whole reason the pair exists.
            assert_eq!(ResidentFormat::of(allocation).allocation(), allocation);
        }
        for layout in TexelLayout::ALL {
            let f = ResidentFormat::of(vk_texel_layout(*layout));
            assert_eq!(f.allocation(), f.declared(), "{layout:?}");
            assert!(!f.needs_own_view(), "{layout:?}");
        }
    }

    /// Sampled byte layouts remain linear while colour attachments retain the
    /// transfer function in their Vulkan format.
    #[test]
    fn the_srgb_rails_still_answer_their_linear_sibling() {
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_BGRA8_UNORM_SRGB).unwrap().0,
            TexelLayout::Bgra8
        );
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_RGBA8_UNORM_SRGB).unwrap().0,
            TexelLayout::Rgba8
        );
        assert_eq!(
            color_attachment(p::MTL_FORMAT_BGRA8_UNORM_SRGB).unwrap().0,
            vk::Format::B8G8R8A8_SRGB
        );
        assert_eq!(
            color_attachment(p::MTL_FORMAT_RGBA8_UNORM_SRGB).unwrap().0,
            vk::Format::R8G8B8A8_SRGB
        );
        // …and each one hands back the decline that loss owes, so the hold is
        // measured rather than assumed.
        for mtl in [
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
        ] {
            assert_eq!(
                sampled_pixels(mtl).unwrap().1,
                Some(TranslateReason::SrgbDowngraded(mtl))
            );
            assert_eq!(color_attachment(mtl).unwrap().1, None);
            // The faithful format is one field away and costs the same bytes.
            let f = translate(mtl).unwrap();
            assert_ne!(f.vk, f.linear_vk);
            assert_eq!(bytes_per_texel(f.vk), bytes_per_texel(f.linear_vk));
        }
    }

    /// The engine rails carry exactly the layouts they are built for, and the
    /// rest decline with a slug that says *this rail*, not *unknown format* —
    /// two causes a reader must be able to tell apart.
    #[test]
    fn a_rail_that_carries_no_layout_declines_with_its_own_slug() {
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_RGBA16_UINT).unwrap_err(),
            TranslateReason::NoSampledLayout(p::MTL_FORMAT_RGBA16_UINT)
        );
        // Its float sibling *is* carried, and at its own width. The two are
        // asserted together because they are one bit depth apart on the wire
        // and it is the pair that says the decline above is about the layout
        // this rail carries rather than about sixteen-bit texels.
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_RGBA16_FLOAT).unwrap().0,
            TexelLayout::Rgba16Float
        );
        assert_eq!(
            TexelLayout::Rgba16Float.bytes_per_texel(),
            crate::contract::pixel_format::RGBA16F_BPP
        );
        assert_eq!(
            sampled_pixels(0xffff).unwrap_err(),
            TranslateReason::UnknownPixelFormat(0xffff)
        );
        // A8Unorm is admitted *with a plan*, not declined. It is one byte like
        // R8Unorm and rides in the same Vulkan format, and the plan is the only
        // thing that distinguishes them: without it the shader gets
        // `(a,0,0,1)` where Metal gives `(0,0,0,a)`.
        let (a8_layout, _, a8_plan) =
            sampled_pixels(p::MTL_FORMAT_A8_UNORM).expect("A8Unorm is sampled, with a plan");
        assert_eq!(a8_layout, TexelLayout::R8);
        assert_eq!(a8_plan, ALPHA_IN_RED);
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_R8_UNORM).unwrap().2,
            IDENTITY,
            "R8Unorm is the same layout and the same Vulkan format as A8Unorm;              only the plan tells them apart"
        );
        assert!(sampled_pixels(p::MTL_FORMAT_R8_UNORM).is_ok());
        // `R8_UNORM` is renderable as of the macos-26 coverage-layer reading, so
        // the "sampled but not a colour attachment" case is carried by a format
        // that still is one. `R32_FLOAT` has a sampled layout (the colour-LUT
        // rail) and no render-target width, which is the pair this asserts:
        // having a layout is not being a colour attachment.
        assert_eq!(
            color_attachment(p::MTL_FORMAT_R32_FLOAT).unwrap_err(),
            TranslateReason::NoColorAttachmentFormat(p::MTL_FORMAT_R32_FLOAT)
        );
        assert_eq!(
            color_attachment(0xffff).unwrap_err(),
            TranslateReason::UnknownPixelFormat(0xffff)
        );
        assert_ne!(
            TranslateReason::NoSampledLayout(0).slug(),
            TranslateReason::UnknownPixelFormat(0).slug()
        );
    }

    /// A single-channel `float16` texture samples natively as `R16_SFLOAT`
    /// (its linear-filter feature is spec-mandatory, so it needs no capability
    /// gate). The color-management LUTs of macOS WindowServer's
    /// `UberCompositeFragment` display-profile pass arrive this way; before this
    /// rail carried the layout the draw resolved to nothing and the whole
    /// color-managed desktop composite failed with `draw_vk_nothing_stored`.
    #[test]
    fn single_channel_float_samples_natively_through_its_own_layout() {
        use crate::contract::pixel_format::TexelLayout;
        let (layout, decline, _) =
            sampled_pixels(p::MTL_FORMAT_R16_FLOAT).expect("R16F is sampled");
        assert_eq!(layout, TexelLayout::R16Float);
        assert!(decline.is_none(), "no sRGB transfer function to drop");
        assert_eq!(layout.bytes_per_texel(), 2);
        assert!(!layout.is_four_byte_color());
        assert_eq!(vk_texel_layout(layout), vk::Format::R16_SFLOAT);
        // R32F names its layout here (a decode fact); the *runtime* rail gates
        // it on the optional linear-filter capability. Four bytes wide but not a
        // colour order, so it must stay out of `is_four_byte_color`.
        let (r32, _, _) = sampled_pixels(p::MTL_FORMAT_R32_FLOAT).expect("R32F is sampled");
        assert_eq!(r32, TexelLayout::R32Float);
        assert_eq!(r32.bytes_per_texel(), 4);
        assert!(!r32.is_four_byte_color());
        assert_eq!(vk_texel_layout(r32), vk::Format::R32_SFLOAT);
    }

    /// Every layout uploads as a Vulkan format exactly as wide as the stride
    /// this device reads its rows at.
    ///
    /// [`vk_texel_layout`] is the one crossing from the decode vocabulary to
    /// the host one, and the two sides carry the width independently: the guest
    /// side is [`TexelLayout::bytes_per_texel`], which every row loader
    /// multiplies by, and the host side is whatever the `vk::Format` occupies.
    /// A disagreement is not a validation error — Vulkan will happily consume
    /// the buffer — it is a sheared or truncated image, which is the failure
    /// mode hardest to attribute from a screenshot.
    ///
    /// Two of the six were pinned by
    /// `single_channel_float_samples_natively_through_its_own_layout`, which is
    /// where the asymmetry was noticed; this is the same check over all six,
    /// with the widths spelled out for the reason
    /// `storage_texel_width_matches_the_pixel_table` gives — a change to
    /// `bytes_per_texel` that silently redefined a stride is exactly what a
    /// derived expectation would fail to catch.
    #[test]
    fn every_texel_layout_uploads_as_a_format_of_its_own_width() {
        use crate::contract::pixel_format::TexelLayout;
        for (layout, format, width) in [
            (TexelLayout::Rgba8, vk::Format::R8G8B8A8_UNORM, 4u32),
            (TexelLayout::Bgra8, vk::Format::B8G8R8A8_UNORM, 4),
            (TexelLayout::R8, vk::Format::R8_UNORM, 1),
            (TexelLayout::Rg8, vk::Format::R8G8_UNORM, 2),
            (TexelLayout::R16Float, vk::Format::R16_SFLOAT, 2),
            (TexelLayout::R32Float, vk::Format::R32_SFLOAT, 4),
            (TexelLayout::R16Unorm, vk::Format::R16_UNORM, 2),
            (TexelLayout::Rg16Unorm, vk::Format::R16G16_UNORM, 4),
        ] {
            assert_eq!(
                vk_texel_layout(layout),
                format,
                "{layout:?} changed the Vulkan format it uploads as"
            );
            assert_eq!(
                layout.bytes_per_texel(),
                width,
                "{layout:?} reads rows at a stride its upload format does not have"
            );
            // The third holder of this width is `bytes_per_texel`, which sizes
            // the linear buffer a sampled image is validated against. A format
            // missing from it is a refused draw rather than a sheared one, and
            // that is how the ten-bit video planes surfaced: admitting them to
            // the layout enum moved the refusal here instead of removing it.
            assert_eq!(
                bytes_per_texel(format),
                Some(width),
                "{format:?} has no linear texel footprint, so a sampled draw \
                 binding it is refused"
            );
        }
    }

    /// [`EXPECTED`] names every format [`translate`] accepts.
    ///
    /// Every other test here iterates `EXPECTED`, so a format added to
    /// `translate` and not to `EXPECTED` is simply never swept — its texel
    /// width, its transfer function, its channel plan and its membership of
    /// each rail's accepted set all go unchecked, and every test still passes.
    /// `MTLPixelFormatRGBA16Unorm` was added to `translate` in the same commit
    /// as this test, and without it nothing would have noticed either way.
    ///
    /// Swept over the whole `u16` domain rather than over a list of constants,
    /// because a list is the thing being checked. This is a derivation, not a
    /// second spelling: `translate` is the authority on what it accepts and it
    /// is asked about every value it could be given.
    #[test]
    fn expected_names_every_format_the_table_translates() {
        let listed: std::collections::BTreeSet<u16> =
            EXPECTED.iter().map(|(mtl, ..)| *mtl).collect();
        let translated: std::collections::BTreeSet<u16> =
            (0..=u16::MAX).filter(|mtl| translate(*mtl).is_ok()).collect();
        assert_eq!(
            translated, listed,
            "translate accepts formats EXPECTED does not name (or the reverse), so they are \
             swept by no test in this module"
        );
    }

    /// Every rail's accepted set, spelled out. A format silently joining or
    /// leaving one of these changes which draws take the zero-copy path.
    #[test]
    fn the_engine_rails_accept_exactly_these_formats() {
        let sampled: Vec<u16> = EXPECTED
            .iter()
            .filter(|(mtl, ..)| sampled_pixels(*mtl).is_ok())
            .map(|(mtl, ..)| *mtl)
            .collect();
        assert_eq!(
            sampled,
            vec![
                // A8Unorm is present, and it is the one format here admitted
                // with a non-identity plan: its byte rides in `R8_UNORM` and
                // the plan puts it back in alpha. Binding it without the plan
                // would hand the shader the byte in red.
                p::MTL_FORMAT_A8_UNORM,
                p::MTL_FORMAT_R8_UNORM,
                // Single-channel float rides its own native rail (color LUTs).
                // Both layouts are named here; the runtime gates R32F on the
                // optional linear-filter capability, but the decode contract
                // itself carries both.
                p::MTL_FORMAT_R16_FLOAT,
                p::MTL_FORMAT_RG8_UNORM,
                p::MTL_FORMAT_R32_FLOAT,
                // The two half-float colour layouts. A recent macOS window
                // server composites in RGBA16Float; before these were named,
                // every such bind fell to the CPU rung and was quantized to
                // unorm8 with everything above 1.0 clamped.
                p::MTL_FORMAT_RG16_FLOAT,
                p::MTL_FORMAT_RGBA8_UNORM,
                p::MTL_FORMAT_RGBA8_UNORM_SRGB,
                p::MTL_FORMAT_BGRA8_UNORM,
                p::MTL_FORMAT_BGRA8_UNORM_SRGB,
                // The packed 32-bit colour family, whose channel boundaries are
                // not byte boundaries — so the CPU rung could not have served
                // them at all and the refusal was the whole loss. `BGR10A2Unorm`
                // is the member a guest was measured binding.
                p::MTL_FORMAT_RGB10A2_UNORM,
                p::MTL_FORMAT_RG11B10_FLOAT,
                p::MTL_FORMAT_BGR10A2_UNORM,
                // The BC block-compressed families, in `EXPECTED`'s order.
                // Named unconditionally here because `sampled_pixels` is a
                // decode fact: whether this host can sample one is
                // `engine::supports_block_compressed_sampled`, which the rail
                // carries in `NativeUploads::block_compressed`. A host without
                // the feature refuses the bind by name — it does not make the
                // format untranslatable.
                p::MTL_FORMAT_BC1_RGBA,
                p::MTL_FORMAT_BC1_RGBA_SRGB,
                p::MTL_FORMAT_BC2_RGBA,
                p::MTL_FORMAT_BC2_RGBA_SRGB,
                p::MTL_FORMAT_BC3_RGBA,
                p::MTL_FORMAT_BC3_RGBA_SRGB,
                p::MTL_FORMAT_BC4_R_UNORM,
                p::MTL_FORMAT_BC4_R_SNORM,
                p::MTL_FORMAT_BC5_RG_UNORM,
                p::MTL_FORMAT_BC5_RG_SNORM,
                p::MTL_FORMAT_BC6H_RGB_FLOAT,
                p::MTL_FORMAT_BC6H_RGB_UFLOAT,
                p::MTL_FORMAT_BC7_RGBA_UNORM,
                p::MTL_FORMAT_BC7_RGBA_UNORM_SRGB,
                // The ten-bit biplanar video planes and the four-channel
                // sixteen-bit unorm. These three were carried by `translate`
                // and by the layout table but were absent from `EXPECTED`, so
                // this list never named them; `expected_names_every_format_
                // the_table_translates` is what surfaced that.
                p::MTL_FORMAT_R16_UNORM,
                p::MTL_FORMAT_RG16_UNORM,
                p::MTL_FORMAT_RGBA16_UNORM,
                p::MTL_FORMAT_RGBA16_FLOAT,
            ]
        );
        let color: Vec<u16> = EXPECTED
            .iter()
            .filter(|(mtl, ..)| color_attachment(*mtl).is_ok())
            .map(|(mtl, ..)| *mtl)
            .collect();
        assert_eq!(
            color,
            vec![
                // macOS 26 renders into a single-channel half-float linear GVA
                // target — a blur/backdrop intermediate — and it was refused as
                // `rt_resolve reason=rt_linear_format` three times a driven
                // boot until `rgba8_to_texel` gained the arm its CPU Store
                // needed. It has been in the sampled list above throughout,
                // which is what made the target renderable-and-readable rather
                // than write-only.
                //
                // `R8_UNORM` is the same reading one format over — a
                // single-channel *eight-bit* linear GVA target, a coverage or
                // mask layer, refused once a driven boot as `fmt=0xa`. It
                // needed three conversion arms rather than one because a
                // one-byte texel had never been a render target here.
                p::MTL_FORMAT_R8_UNORM,
                p::MTL_FORMAT_R16_FLOAT,
                p::MTL_FORMAT_RG16_FLOAT,
                p::MTL_FORMAT_RGBA8_UNORM,
                p::MTL_FORMAT_RGBA8_UNORM_SRGB,
                p::MTL_FORMAT_BGRA8_UNORM,
                p::MTL_FORMAT_BGRA8_UNORM_SRGB,
                // The first packed 32-bit colour attachment, and the first one
                // admitted for a *game* rather than for the window server. An
                // `'l10r'` IOSurface — `kCVPixelFormatType_ARGB2101010LEPacked`
                // — is what Asphalt 8 renders into on a macos-13 x86/Vulkan
                // boot, and every draw of it failed at
                // `draw::render_target`'s `rt_type4_base_format` until the
                // FourCC and `render_target_bpp` both named it.
                //
                // Unlike every other member here this one is not a format
                // Vulkan mandates for `COLOR_ATTACHMENT_BIT`, so a host may
                // decline it and this rail is where that decline appears. The
                // NVIDIA host it was measured on advertises it.
                p::MTL_FORMAT_BGR10A2_UNORM,
                // Admitted since the two arms became one answer. The contract
                // has said a half-float render target is renderable since one
                // could be created at that format; only this side still refused
                // it, so a half-float secondary MRT slot was declined while a
                // half-float primary was not.
                p::MTL_FORMAT_RGBA16_FLOAT,
            ]
        );
    }

    /// An integer texel is declared, translates, and is refused by every rail
    /// that would have to give it a meaning — each by its own name.
    ///
    /// The refusal is the point. `R8Uint` holds an eight-bit *integer*, and every
    /// converter in `pixel_format` reads a one-byte texel as a unorm: run through
    /// them, a stored 200 comes back as 0.784 and the shader is handed a number
    /// the guest never wrote. So the correct rail for an integer format is the
    /// native one or none, and until a guest is measured needing the native one,
    /// none is the honest answer.
    ///
    /// What the declaration bought is a *precise* refusal rather than a
    /// misleading one: before it, `bytes_per_pixel` answered `None` and the bind
    /// died at the width gate as `format_incompatible`, which names a guest
    /// error. Now each rail that cannot take it says so about itself.
    #[test]
    fn an_integer_texel_is_declared_but_has_no_sampled_rail() {
        // Both members macOS 26 stages, and the second was found only by
        // admitting the first: one dispatch binds both, so the refusal moved
        // from `0x0d` to `0x21` at an unchanged count.
        let integers: &[(u16, vk::Format, u32)] = &[
            (p::MTL_FORMAT_R8_UINT, vk::Format::R8_UINT, p::R8_BPP),
            (p::MTL_FORMAT_RG8_UINT, vk::Format::R8G8_UINT, p::RG8_BPP),
        ];
        for &(mtl, vk_format, bpp) in integers {
            // Declared: it has a width and a Vulkan spelling.
            assert_eq!(p::bytes_per_pixel(mtl), Some(bpp), "{mtl:#x} texel width");
            assert_eq!(translate(mtl).unwrap().vk, vk_format);

            // Refused, and each by its own name rather than by a shared slug.
            assert!(
                matches!(sampled_pixels(mtl), Err(TranslateReason::NoSampledLayout(f)) if f == mtl),
                "{mtl:#x} must decline the sampled rail by name"
            );
            assert!(color_attachment(mtl).is_err());
            assert_eq!(p::render_target_bpp(mtl), None);
            assert_eq!(p::storage_selector(mtl), None);

            // And it never reaches a unorm converter: no texel layout means no
            // conversion arm can silently claim it.
            assert_eq!(
                texel_layout_of(vk_format),
                None,
                "{mtl:#x} must have no guest texel layout"
            );
        }
    }

    /// The two arms that answer "may a colour attachment be this format" are one
    /// answer, and every format they admit can survive both readback rails.
    ///
    /// They were two hand-kept lists in two vocabularies — `render_target_bpp`
    /// over `MTLPixelFormat`, this one over `vk::Format` — so nothing could
    /// compare them, and they had drifted: the contract admitted
    /// `RGBA16_FLOAT`, which is what lets a half-float *primary* attachment be
    /// created at the format the guest declared, while `color_attachment`
    /// refused it. The same guest format was renderable as slot 0 and declined
    /// as a secondary MRT slot.
    ///
    /// The second half is the obligation `render_target_bpp`'s doc states.
    /// A renderable format whose layout cannot narrow to RGBA8 is a target the
    /// readback rails lose the frame of, and one that cannot expand from RGBA8
    /// is a target whose CPU `Load` seed is refused — both silent until a guest
    /// asks for that format. `Rg16Float` was admitted and could do neither for
    /// as long as it had been renderable.
    #[test]
    fn the_renderable_set_is_one_answer_and_every_member_survives_both_rails() {
        for &(mtl, ..) in EXPECTED {
            let admitted = color_attachment(mtl).is_ok();
            assert_eq!(
                admitted,
                p::render_target_bpp(mtl).is_some(),
                "{mtl:#x}: the two colour-attachment arms disagree"
            );
            if !admitted {
                continue;
            }
            let format = color_attachment(mtl).unwrap().0;
            // Readback moves stored texels and therefore reasons about the
            // linear sibling's byte layout; an sRGB image view changes the
            // shader conversion, not those bytes.
            let storage_format = translate(mtl).unwrap().linear_vk;
            let layout = texel_layout_of(storage_format).unwrap_or_else(|| {
                panic!("{mtl:#x}: renderable as {format:?} with no guest texel layout")
            });
            // Four pixels of each, through both directions. The functions check
            // their own lengths, so a `false` here is the layout being unhandled
            // rather than a short buffer.
            const PX: u32 = 4;
            let wide = vec![0u8; PX as usize * layout.bytes_per_texel() as usize];
            let mut rgba = vec![0u8; PX as usize * p::RGBA8_BPP as usize];
            assert!(
                p::narrow_texel_to_rgba8(layout, &wide, PX, &mut rgba),
                "{mtl:#x}: renderable as {layout:?}, which no readback rail can narrow"
            );
            let mut back = wide.clone();
            assert!(
                p::expand_rgba8_to_texel(layout, &rgba, PX, &mut back),
                "{mtl:#x}: renderable as {layout:?}, which no CPU Load seed can expand to"
            );
            // The third rail, and the one whose gap is a lost frame rather than
            // a slow one. When the GPU cannot land a Store in guest pages the
            // synchronous Store reads the resident back and converts it row by
            // row into the guest's declared format — so a renderable format this
            // refuses renders fine and then loses every frame on any host
            // without a guest-RAM import. `R16_FLOAT` was exactly that gap.
            let mut row = vec![0u8; PX as usize * p::bytes_per_pixel(mtl).unwrap() as usize];
            assert!(
                p::convert_rgba8_to_row(mtl, &rgba, PX, &mut row),
                "{mtl:#x}: renderable, and the CPU Store converter cannot write it"
            );
        }
    }

    /// Every guest texel layout has a Vulkan-side width, and it is the contract's.
    ///
    /// This is the check that was missing. `bytes_per_texel` used to be a second,
    /// hand-kept copy of `TexelLayout::bytes_per_texel`, and when `Rgba16Unorm`
    /// was added to the contract nothing made this side learn it. A `None` here
    /// is not a quiet wrong answer — it is the same verdict a block-compressed
    /// format gets, so the draw is refused by name
    /// (`vk_draw_validate_sampled_no_linear_texel_footprint`) and the guest
    /// silently loses it. macOS 26 lost 108 draws a boot to exactly that.
    ///
    /// Asserting equality with the contract rather than a literal is the point:
    /// a literal table here is what created the drift in the first place.
    #[test]
    fn every_texel_layout_has_the_contract_width_on_the_vulkan_side() {
        for &layout in TexelLayout::ALL {
            let vk = vk_texel_layout(layout);
            assert_eq!(
                bytes_per_texel(vk),
                Some(layout.bytes_per_texel()),
                "{layout:?} ({vk:?}) disagrees with the contract width"
            );
        }
    }

    /// A format that is genuinely not one texel-width answers `None`, so the
    /// derivation above did not turn the decline into a wrong number.
    #[test]
    fn a_block_compressed_format_still_has_no_texel_footprint() {
        assert_eq!(bytes_per_texel(vk::Format::BC1_RGB_UNORM_BLOCK), None);
        assert_eq!(bytes_per_texel(vk::Format::G8_B8R8_2PLANE_420_UNORM), None);
    }

    /// The engine-internal format constants are not a second opinion: each is
    /// exactly what the pixel table answers for the Metal format it stands for.
    /// A drift here is a red/blue channel swap on the present path, which reads
    /// as a rendering bug rather than a translation one.
    #[test]
    fn the_engine_format_constants_come_from_the_table() {
        assert_eq!(
            SCANOUT_FORMAT,
            translate(p::MTL_FORMAT_BGRA8_UNORM).unwrap().vk
        );
        assert_eq!(
            RESIDENT_RGBA_FORMAT,
            translate(p::MTL_FORMAT_RGBA8_UNORM).unwrap().vk
        );
        assert_eq!(
            TRANSIENT_DEPTH_FORMAT,
            translate(p::MTL_FORMAT_DEPTH32_FLOAT).unwrap().vk
        );
        assert_eq!(resident_color(true), SCANOUT_FORMAT);
        assert_eq!(resident_color(false), RESIDENT_RGBA_FORMAT);
        assert_ne!(SCANOUT_FORMAT, RESIDENT_RGBA_FORMAT);
    }

    /// A sampled bind's view mapping is the decoded type-8 swizzle and nothing
    /// else. Identity in, identity out — otherwise every ordinary bind would
    /// pay for a feature almost none of them use.
    #[test]
    fn a_sampled_bind_maps_identity_to_identity() {
        let m = vk_component_mapping(&pixel_format::swizzle_identity());
        assert_eq!(m.r, vk::ComponentSwizzle::R);
        assert_eq!(m.g, vk::ComponentSwizzle::G);
        assert_eq!(m.b, vk::ComponentSwizzle::B);
        assert_eq!(m.a, vk::ComponentSwizzle::A);
    }

    /// A non-identity view reaches the hardware unchanged — the swizzle is a
    /// property of the view, not of the byte order underneath it.
    #[test]
    fn a_sampled_bind_carries_the_view_swizzle() {
        // Read `(b, g, r, 1)`.
        let view = pixel_format::swizzle_plan(&[4, 3, 2, 1]).unwrap();
        let m = vk_component_mapping(&view);
        assert_eq!(m.r, vk::ComponentSwizzle::B);
        assert_eq!(m.g, vk::ComponentSwizzle::G);
        assert_eq!(m.b, vk::ComponentSwizzle::R);
        assert_eq!(m.a, vk::ComponentSwizzle::ONE);
    }

    /// The storage rail *declines* sRGB rather than downgrading it, which is
    /// why it keeps a layout enum where the colour and sampled rails now
    /// resolve to a `VkFormat`.
    ///
    /// Pins the load-bearing half of that argument: widening this vocabulary
    /// could not make a colour space newly reachable, because no sRGB format
    /// gets through the contract's storage selector in the first place. If one
    /// ever does, this fails and the decision is up for review — the rail would
    /// then be silently dropping a transfer function with no census site
    /// watching it.
    #[test]
    fn no_srgb_format_reaches_the_storage_rail() {
        let mut checked = 0;
        for &(mtl, ..) in EXPECTED {
            if !is_srgb(mtl) {
                continue;
            }
            checked += 1;
            assert_eq!(
                storage_image(mtl).unwrap_err(),
                TranslateReason::NoStorageImageFormat(mtl),
                "MTL {mtl:#x} is sRGB and reached the storage rail"
            );
        }
        assert!(checked >= 2, "the table lists no sRGB formats to check");

        // …and nothing the rail *does* admit carries a transfer function, so
        // `PixelFormat::vk` and `linear_vk` coincide for every one of them.
        for &(mtl, ..) in EXPECTED {
            if storage_image(mtl).is_ok() {
                let f = translate(mtl).unwrap();
                assert_eq!(f.vk, f.linear_vk, "MTL {mtl:#x}");
            }
        }
    }

    /// The invariant the sampled rail's view mapping rests on, checked against
    /// the table rather than a hand-listed set of layouts.
    ///
    /// The pool binds `vk_component_mapping(view)` with no contribution from
    /// the format, which is correct only while every format reaching that rail
    /// has an identity component plan. `A8Unorm` is the one that does not, and
    /// it must be declined rather than admitted — binding it as plain
    /// `R8_UNORM` would hand the shader `(a,0,0,1)` where Metal gives
    /// `(0,0,0,a)`.
    #[test]
    fn every_format_the_sampled_rail_admits_reports_its_own_mapping() {
        let mut admitted_any = false;
        let mut saw_non_identity = false;
        for &(mtl, ..) in EXPECTED {
            let Ok((_, _, plan)) = sampled_pixels(mtl) else {
                continue;
            };
            admitted_any = true;
            // The plan handed back must be the format's own, not a default. A
            // rail that folds it into its view mapping is only correct if this
            // is the same answer `translate` gives.
            assert_eq!(
                plan,
                translate(mtl).unwrap().components,
                "sampled_pixels reports a different plan for MTLPixelFormat {mtl:#x} than \
                 the format table does"
            );
            saw_non_identity |= plan != IDENTITY;
            assert_eq!(has_identity_components(mtl), plan == IDENTITY);
        }
        assert!(admitted_any, "the sampled rail admits nothing at all");
        // This used to assert the opposite — that every admitted format needed
        // no mapping — which was true only because the one that does was
        // refused. It is admitted now, and a suite where no admitted format has
        // a non-identity plan would silently stop testing the fold.
        assert!(
            saw_non_identity,
            "no admitted format carries a mapping, so the composition at the bind site is untested"
        );
    }

    /// The typed reason and the always-on census line must name the class
    /// identically, or a grep of the fail log misses half the evidence.
    #[test]
    fn the_downgrade_slug_matches_the_always_on_census() {
        assert_eq!(
            TranslateReason::SrgbDowngraded(0).slug(),
            crate::runtime::census::srgb_census::SRGB_DOWNGRADED_SLUG
        );
    }
}
