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
    self, SwizzlePlan, SwizzleSource, TexelLayout, COMPONENT_A, COMPONENT_B, COMPONENT_G,
    COMPONENT_R,
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
/// *contract* question answered by `render_target_class` / `storage_selector` /
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
        p::MTL_FORMAT_R16_UNORM => linear(vk::Format::R16_UNORM, 2),
        p::MTL_FORMAT_RG16_UNORM => linear(vk::Format::R16G16_UNORM, 4),
        p::MTL_FORMAT_RG16_UINT => linear(vk::Format::R16G16_UINT, 4),
        p::MTL_FORMAT_R16_FLOAT => linear(vk::Format::R16_SFLOAT, 2),
        p::MTL_FORMAT_RG8_UNORM => linear(vk::Format::R8G8_UNORM, 2),
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
pub fn sampled_pixels(mtl: u16) -> Result<(TexelLayout, Option<TranslateReason>), TranslateReason> {
    let f = translate(mtl)?;
    // A format whose Metal channels do not sit identically on its Vulkan
    // channels needs a component mapping on the view to sample correctly.
    // `TexelLayout` names a byte layout only and carries no mapping, so
    // admitting such a format here would bind, say, `A8Unorm` as plain
    // `R8_UNORM` and hand the shader `(a,0,0,1)` where Metal gives `(0,0,0,a)`.
    // Decline until the rail carries the mapping.
    if f.components != IDENTITY {
        return Err(TranslateReason::NoSampledLayout(mtl));
    }
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
        vk::Format::R16_UNORM => TexelLayout::R16Unorm,
        vk::Format::R16G16_UNORM => TexelLayout::Rg16Unorm,
        vk::Format::R16G16_UINT => TexelLayout::Rg16Uint,
        vk::Format::R16_SFLOAT => TexelLayout::R16Float,
        vk::Format::R32_SFLOAT => TexelLayout::R32Float,
        _ => return Err(TranslateReason::NoSampledLayout(mtl)),
    };
    Ok((layout, srgb_decline(&f, mtl)))
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
        TexelLayout::R16Unorm => vk::Format::R16_UNORM,
        TexelLayout::Rg16Unorm => vk::Format::R16G16_UNORM,
        TexelLayout::Rg16Uint => vk::Format::R16G16_UINT,
        TexelLayout::R16Float => vk::Format::R16_SFLOAT,
        TexelLayout::R32Float => vk::Format::R32_SFLOAT,
    }
}

/// Every Vulkan format a colour attachment may take, and the decline for a
/// format the rail does not render to.
///
/// The result is the resolved [`vk::Format`] rather than an engine enum, so an
/// sRGB target is *expressible* here — the render pass, the pipeline key and
/// the image all carry a real format now. It is still [`PixelFormat::linear_vk`]
/// that is returned, with the [`TranslateReason::SrgbDowngraded`] that loss
/// owes: flipping the rail is a separate, measurable change, because the crate
/// currently ignores the transfer function *consistently* and a target written
/// unencoded then sampled undecoded cancels out.
///
/// The narrowing is deliberate and stays. Metal renders to far more formats
/// than these three; admitting one the rest of the pass machinery has never
/// carried would trade a named decline for a wrong picture.
pub fn color_attachment(
    mtl: u16,
) -> Result<(vk::Format, Option<TranslateReason>), TranslateReason> {
    let f = translate(mtl)?;
    if !matches!(
        f.linear_vk,
        vk::Format::R8G8B8A8_UNORM
            | vk::Format::B8G8R8A8_UNORM
            | vk::Format::R16G16_SFLOAT
            | vk::Format::R16G16B16A16_SFLOAT
    ) {
        return Err(TranslateReason::NoColorAttachmentFormat(mtl));
    }
    Ok((f.linear_vk, srgb_decline(&f, mtl)))
}

/// The engine's storage-image format for a contract [`StorageImageSelector`]
/// ordinal.
///
/// The selector is the compute rail's own narrowing of `MTLPixelFormat`, so
/// this is a vocabulary-to-vocabulary step rather than a Metal decision — but
/// it lives here for the same reason everything else does: it was previously
/// spelled in `runtime/compute_exec/mod.rs`, where nothing could see that the two
/// enums had to stay in step. A selector with no engine format means the
/// vocabularies have drifted, which is a different failure from a format the
/// rail does not support, so it declines by its own name.
pub fn storage_image_from_selector(selector: u32) -> Result<StorageImageFormat, TranslateReason> {
    use crate::contract::pixel_format::StorageImageSelector as S;
    Ok(match selector {
        s if s == S::Rgba8Uint as u32 => StorageImageFormat::Rgba8Uint,
        s if s == S::Rgba8Sint as u32 => StorageImageFormat::Rgba8Sint,
        s if s == S::Rgba16Uint as u32 => StorageImageFormat::Rgba16Uint,
        s if s == S::Rgba16Float as u32 => StorageImageFormat::Rgba16Float,
        s if s == S::Rgba32Float as u32 => StorageImageFormat::Rgba32Float,
        s if s == S::Rgba8Unorm as u32 => StorageImageFormat::Rgba8Unorm,
        s if s == S::Bgra8Unorm as u32 => StorageImageFormat::Bgra8Unorm,
        s if s == S::R16Float as u32 => StorageImageFormat::R16Float,
        s if s == S::Rg16Float as u32 => StorageImageFormat::Rg16Float,
        s if s == S::R8Unorm as u32 => StorageImageFormat::R8Unorm,
        s if s == S::Rg8Unorm as u32 => StorageImageFormat::Rg8Unorm,
        s if s == S::Rgba32Uint as u32 => StorageImageFormat::Rgba32Uint,
        s if s == S::R32Uint as u32 => StorageImageFormat::R32Uint,
        other => return Err(TranslateReason::UnknownStorageSelector(other)),
    })
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
        pf::MTL_FORMAT_R16_UNORM => return Ok(StorageImageFormat::R16Unorm),
        pf::MTL_FORMAT_RG16_UNORM => return Ok(StorageImageFormat::Rg16Unorm),
        pf::MTL_FORMAT_RG16_UINT => return Ok(StorageImageFormat::Rg16Uint),
        _ => {}
    }
    let selector = pf::storage_selector(mtl).ok_or(TranslateReason::NoStorageImageFormat(mtl))?;
    storage_image_from_selector(selector as u32)
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
pub fn bytes_per_texel(format: vk::Format) -> Option<u32> {
    Some(match format {
        vk::Format::R8_UNORM | vk::Format::S8_UINT => 1,
        vk::Format::R8G8_UNORM
        | vk::Format::R16_SFLOAT
        | vk::Format::R16_UNORM
        | vk::Format::D16_UNORM => 2,
        vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R32_SFLOAT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_UINT
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::E5B9G9R9_UFLOAT_PACK32
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT => 4,
        vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::D32_SFLOAT_S8_UINT => 8,
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
        StorageImageFormat::Rg16Uint => vk::Format::R16G16_UINT,
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
/// The plan passed in is the decoded type-8 view swizzle and nothing else — a
/// format's own channel remap is not composed into it. That is safe because
/// [`sampled_pixels`] declines every format whose plan is not identity, which
/// `every_format_the_sampled_rail_admits_needs_no_mapping_of_its_own` holds it
/// to; `A8Unorm` is the only such format and it is refused, not reshaped.
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
/// The sampled rail relies on that: its view mapping is the decoded type-8
/// swizzle alone (see [`vk_component_mapping`]), which is only correct because
/// nothing with a non-identity plan reaches it. A test holds the two in
/// agreement.
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
            p::MTL_FORMAT_RGB9E5_FLOAT,
            vk::Format::E5B9G9R9_UFLOAT_PACK32,
            4,
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
    #[test]
    fn texel_size_agrees_with_the_decode_contract() {
        for (mtl, _, _, _) in EXPECTED {
            assert_eq!(
                Some(translate(*mtl).unwrap().bytes_per_texel),
                pixel_format::bytes_per_pixel(*mtl),
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

    /// **The phase-2 regression gate.** An sRGB Metal format may reach a linear
    /// engine format only together with a recorded decline — that is the whole
    /// difference between a downgrade and the silent fold this refactor exists
    /// to remove. Asserted over every entry point that can lose the qualifier.
    #[test]
    fn an_srgb_format_never_reaches_a_linear_one_silently() {
        for mtl in [
            p::MTL_FORMAT_RGBA8_UNORM_SRGB,
            p::MTL_FORMAT_BGRA8_UNORM_SRGB,
        ] {
            let (_, decline) = sampled_pixels(mtl).unwrap();
            assert_eq!(
                decline,
                Some(TranslateReason::SrgbDowngraded(mtl)),
                "sampled rail dropped sRGB with no decline"
            );

            let (_, decline) = color_attachment(mtl).unwrap();
            assert_eq!(
                decline,
                Some(TranslateReason::SrgbDowngraded(mtl)),
                "colour attachment dropped sRGB with no decline"
            );
        }
        // The converse: a linear format must never produce the decline, or the
        // proxy floods and stops meaning anything.
        for (mtl, _, _, transfer) in EXPECTED {
            if *transfer == TransferFunction::Srgb {
                continue;
            }
            if let Ok((_, decline)) = sampled_pixels(*mtl) {
                assert_eq!(decline, None, "MTL {mtl:#x}");
            }
            if let Ok((_, decline)) = color_attachment(*mtl) {
                assert_eq!(decline, None, "MTL {mtl:#x}");
            }
        }
    }

    /// An sRGB format resolves to the same byte layout, and to its linear
    /// sibling's Vulkan format, on both rails that hold the transfer function.
    ///
    /// This is the *held* state, not a limitation of the vocabulary: the
    /// colour-attachment rail now answers a real `VkFormat`, so the day the
    /// crate flips, `B8G8R8A8_SRGB` is one word away — and the equalities below
    /// are what will change, deliberately and visibly.
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
            vk::Format::B8G8R8A8_UNORM
        );
        assert_eq!(
            color_attachment(p::MTL_FORMAT_RGBA8_UNORM_SRGB).unwrap().0,
            vk::Format::R8G8B8A8_UNORM
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
            assert_eq!(
                color_attachment(mtl).unwrap().1,
                Some(TranslateReason::SrgbDowngraded(mtl))
            );
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
            sampled_pixels(p::MTL_FORMAT_RGBA16_FLOAT).unwrap_err(),
            TranslateReason::NoSampledLayout(p::MTL_FORMAT_RGBA16_FLOAT)
        );
        assert_eq!(
            sampled_pixels(0xffff).unwrap_err(),
            TranslateReason::UnknownPixelFormat(0xffff)
        );
        // Declined for the *mapping*, not the byte size: A8Unorm is one byte
        // like R8Unorm but does not present its channels the same way.
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_A8_UNORM).unwrap_err(),
            TranslateReason::NoSampledLayout(p::MTL_FORMAT_A8_UNORM)
        );
        assert!(sampled_pixels(p::MTL_FORMAT_R8_UNORM).is_ok());
        assert_eq!(
            color_attachment(p::MTL_FORMAT_R8_UNORM).unwrap_err(),
            TranslateReason::NoColorAttachmentFormat(p::MTL_FORMAT_R8_UNORM)
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
    /// Every layout the sampled rail can bind must have a texel footprint the
    /// draw validator recognises.
    ///
    /// These are two tables in two files. `vk_texel_layout` decides the Vulkan
    /// format a sampled bind carries; `bytes_per_texel` is what
    /// `validate_sampled_images` then measures the CPU-origin buffer against,
    /// and a format missing from it is refused by name — which drops the whole
    /// draw, not the one texture.
    ///
    /// Adding `R16_UNORM` to the first table without the second did exactly
    /// that: the wallpaper's luma plane decoded, bound, and then died at
    /// `vk_draw_validate_sampled_no_linear_texel_footprint binding=163
    /// format=R16_UNORM` on a 1920x1088 full-screen quad. The two tables have to
    /// be checked against each other rather than maintained in parallel by hand.
    ///
    /// `index_in_all` is exhaustive on purpose: a new `TexelLayout` variant
    /// fails to compile there, which forces it into `ALL` and so into this
    /// check.
    #[test]
    fn every_sampled_layout_has_a_texel_footprint_the_validator_knows() {
        use crate::contract::pixel_format::TexelLayout;

        const ALL: &[TexelLayout] = &[
            TexelLayout::Rgba8,
            TexelLayout::Bgra8,
            TexelLayout::R8,
            TexelLayout::Rg8,
            TexelLayout::R16Float,
            TexelLayout::R16Unorm,
            TexelLayout::Rg16Unorm,
            TexelLayout::Rg16Uint,
            TexelLayout::R32Float,
        ];

        fn index_in_all(layout: TexelLayout) -> usize {
            match layout {
                TexelLayout::Rgba8 => 0,
                TexelLayout::Bgra8 => 1,
                TexelLayout::R8 => 2,
                TexelLayout::Rg8 => 3,
                TexelLayout::R16Float => 4,
                TexelLayout::R16Unorm => 5,
                TexelLayout::Rg16Unorm => 6,
                TexelLayout::Rg16Uint => 7,
                TexelLayout::R32Float => 8,
            }
        }

        for (position, layout) in ALL.iter().copied().enumerate() {
            assert_eq!(
                index_in_all(layout),
                position,
                "ALL and index_in_all disagree about {layout:?}"
            );
            let format = vk_texel_layout(layout);
            let footprint = bytes_per_texel(format).unwrap_or_else(|| {
                panic!("{layout:?} binds as {format:?}, which the draw validator cannot size")
            });
            assert_eq!(
                footprint,
                layout.bytes_per_texel(),
                "{layout:?} disagrees with {format:?} about its texel size"
            );
        }
    }

    /// `RG16Uint` samples as integer texels and never rides the colour rails.
    ///
    /// The guest shader settles the type twice over: its decoded argument type
    /// and translated SPIR-V image type both require integer texels. A `_UNORM`
    /// or `_SFLOAT` there would be an invalid descriptor.
    ///
    /// It is also fetch-only — every access is `OpImageFetch` or
    /// `OpImageQuerySizeLod`, never `OpSampledImage` — so it needs no linear
    /// filtering, which Vulkan does not offer for integer formats anyway.
    #[test]
    fn two_channel_uint_samples_as_integer_texels_and_never_as_colour() {
        use crate::contract::pixel_format::{self as p, TexelLayout};

        let (layout, _decline) =
            sampled_pixels(p::MTL_FORMAT_RG16_UINT).expect("RG16Uint is sampled");
        assert_eq!(layout, TexelLayout::Rg16Uint);
        assert_eq!(vk_texel_layout(layout), vk::Format::R16G16_UINT);
        assert_eq!(layout.bytes_per_texel(), 4);

        // Distinct from the other two four-byte two-channel layouts: reading
        // integer texels as unorm or float would be a different image.
        assert_ne!(vk_texel_layout(layout), vk_texel_layout(TexelLayout::Rg16Unorm));
        assert_ne!(vk_texel_layout(layout), vk_texel_layout(TexelLayout::R16Float));

        // Kept off every rail that carries colour through 8-bit unorm LUTs. A
        // 16-bit coordinate does not survive that round trip, and a named
        // refusal is better than wrong pixels.
        assert_eq!(p::render_target_bpp(p::MTL_FORMAT_RG16_UINT), None);
        assert_eq!(p::texel_to_rgba8(p::MTL_FORMAT_RG16_UINT, &[0u8; 4]), None);
        assert!(!p::rgba8_to_texel(
            p::MTL_FORMAT_RG16_UINT,
            [1, 2, 3, 4],
            &mut [0u8; 4]
        ));

        // But its size is known, which is what the sampled bind needs: an
        // unknown width is what made this format read as a bpp mismatch against
        // itself (`base_fmt=0x3f view_fmt=0x3f`).
        assert_eq!(p::bytes_per_pixel(p::MTL_FORMAT_RG16_UINT), Some(4));
    }

    /// `R16Unorm` reaches the GPU as one 16-bit channel, not as two 8-bit ones.
    ///
    /// It has no arm in the CPU `convert_row_to_rgba8` loader, so without a
    /// native rail every bind of such a view is refused — measured as 387
    /// `type5_view_convert` declines of a single 3840x2160 view in one logged-in
    /// macOS session, the largest decline class in that boot.
    ///
    /// The layout it lands on decides how the GPU reads the texel, so the
    /// distinction from the other two-byte layouts is the whole point.
    #[test]
    fn single_channel_unorm_samples_natively_and_not_as_two_bytes() {
        use crate::contract::pixel_format::TexelLayout;

        let (layout, _decline) =
            sampled_pixels(p::MTL_FORMAT_R16_UNORM).expect("R16Unorm is sampled");
        assert_eq!(layout, TexelLayout::R16Unorm);
        assert_eq!(vk_texel_layout(layout), vk::Format::R16_UNORM);
        // Two bytes per texel, like `Rg8` and `R16Float` — but a distinct layout,
        // because `R8G8_UNORM` would read the same bytes as two channels.
        assert_eq!(layout.bytes_per_texel(), 2);
        assert_ne!(layout, TexelLayout::Rg8);
        assert_ne!(vk_texel_layout(layout), vk_texel_layout(TexelLayout::Rg8));
        assert_ne!(vk_texel_layout(layout), vk_texel_layout(TexelLayout::R16Float));
    }

    #[test]
    fn single_channel_float_samples_natively_through_its_own_layout() {
        use crate::contract::pixel_format::TexelLayout;
        let (layout, decline) = sampled_pixels(p::MTL_FORMAT_R16_FLOAT).expect("R16F is sampled");
        assert_eq!(layout, TexelLayout::R16Float);
        assert!(decline.is_none(), "no sRGB transfer function to drop");
        assert_eq!(layout.bytes_per_texel(), 2);
        assert!(!layout.is_four_byte_color());
        assert_eq!(vk_texel_layout(layout), vk::Format::R16_SFLOAT);
        // R32F names its layout here (a decode fact); the *runtime* rail gates
        // it on the optional linear-filter capability. Four bytes wide but not a
        // colour order, so it must stay out of `is_four_byte_color`.
        let (r32, _) = sampled_pixels(p::MTL_FORMAT_R32_FLOAT).expect("R32F is sampled");
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
        }
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
                // A8Unorm is absent on purpose: it needs a component mapping
                // this rail cannot carry, and binding it as bare R8 would hand
                // the shader the byte in red instead of alpha.
                p::MTL_FORMAT_R8_UNORM,
                // Single-channel float rides its own native rail (color LUTs).
                // Both layouts are named here; the runtime gates R32F on the
                // optional linear-filter capability, but the decode contract
                // itself carries both.
                p::MTL_FORMAT_R16_FLOAT,
                p::MTL_FORMAT_RG8_UNORM,
                p::MTL_FORMAT_R32_FLOAT,
                p::MTL_FORMAT_RGBA8_UNORM,
                p::MTL_FORMAT_RGBA8_UNORM_SRGB,
                p::MTL_FORMAT_BGRA8_UNORM,
                p::MTL_FORMAT_BGRA8_UNORM_SRGB,
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
                p::MTL_FORMAT_RG16_FLOAT,
                p::MTL_FORMAT_RGBA8_UNORM,
                p::MTL_FORMAT_RGBA8_UNORM_SRGB,
                p::MTL_FORMAT_BGRA8_UNORM,
                p::MTL_FORMAT_BGRA8_UNORM_SRGB,
                // Tahoe renders its glass material into RGBA16Float: every app
                // icon and every Liquid Glass surface is a target of this
                // format. Refusing it did not degrade those surfaces, it lost
                // them — a blank rounded square where the icon is, and no
                // material at all.
                p::MTL_FORMAT_RGBA16_FLOAT,
            ]
        );
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
    fn every_format_the_sampled_rail_admits_needs_no_mapping_of_its_own() {
        let mut admitted_any = false;
        for &(mtl, ..) in EXPECTED {
            if sampled_pixels(mtl).is_ok() {
                admitted_any = true;
                assert!(
                    has_identity_components(mtl),
                    "sampled_pixels admits MTLPixelFormat {mtl:#x}, whose channels do not sit \
                     identically on its Vulkan format; the sampled view binds the type-8 \
                     swizzle alone and would drop that format's own plan"
                );
            }
        }
        assert!(admitted_any, "the sampled rail admits nothing at all");
        // The known non-identity format is declined, not silently reshaped.
        assert!(!has_identity_components(p::MTL_FORMAT_A8_UNORM));
        assert!(matches!(
            sampled_pixels(p::MTL_FORMAT_A8_UNORM),
            Err(TranslateReason::NoSampledLayout(_))
        ));
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

#[cfg(test)]
mod sampled_only_storage_tests {
    use super::*;
    use crate::contract::pixel_format as pf;

    /// The compute rails stage a sampled texture through the storage-image
    /// enum, so a format with no storage selector still has to resolve here.
    /// Without an answer the whole dispatch declines
    /// (`mtl_format_unsupported`) and the guest's work is lost, which is what
    /// happened to the 16-bit single- and two-channel formats.
    #[test]
    fn sixteen_bit_narrow_formats_resolve_for_sampled_staging() {
        for (mtl, expected, bytes) in [
            (pf::MTL_FORMAT_R16_UNORM, vk::Format::R16_UNORM, 2usize),
            (pf::MTL_FORMAT_RG16_UNORM, vk::Format::R16G16_UNORM, 4),
            (pf::MTL_FORMAT_RG16_UINT, vk::Format::R16G16_UINT, 4),
        ] {
            let resolved = storage_image(mtl)
                .unwrap_or_else(|e| panic!("{mtl:#x} has no storage-image format: {e:?}"));
            assert_eq!(vk_storage_image(resolved), expected, "{mtl:#x}");
            assert_eq!(resolved.bytes_per_texel(), bytes, "{mtl:#x}");
        }
    }
}

#[cfg(test)]
mod rgba16f_attachment_tests {
    use super::*;
    use crate::contract::pixel_format as pf;

    /// `MTLPixelFormatRGBA16Float` is what Tahoe renders its glass material
    /// into — every app icon and every Liquid Glass surface is an RGBA16Float
    /// target. Refusing it as a colour attachment does not degrade those
    /// surfaces, it loses them: the icon is a blank rounded square and the
    /// material never appears.
    #[test]
    fn rgba16_float_is_a_colour_attachment() {
        let (format, _) = color_attachment(pf::MTL_FORMAT_RGBA16_FLOAT)
            .expect("RGBA16Float has no colour attachment format");
        assert_eq!(format, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(bytes_per_texel(format), Some(8));
    }
}
