//! The Vulkan rail's pixel table, as this device's engine and rails ask for it.
//!
//! **The Metal→Vulkan table itself is [`reims_vgpu_vulkan::pixel`].** It is a
//! function of its inputs with no state, no handle and no host capability, so
//! it belongs to the rail crate and not to the device model that calls it. What
//! is left here is the part that names things the rail crate must not: the
//! engine's own [`StorageImageFormat`] vocabulary, the clear-value carrier a
//! colour attachment is created with, and the always-on
//! [`srgb_census`](crate::runtime::census::srgb_census) that records a dropped
//! transfer function.
//!
//! # Who records the loss changed, deliberately
//!
//! The table used to note the sRGB downgrade from inside the translation. The
//! rail crate cannot see `runtime`, and it should not: *which* rail took a
//! downgrade is a fact about the caller, not about the format. So the doors
//! that can lose the qualifier now say so in their answer —
//! [`reims_vgpu_vulkan::pixel::Sampled::srgb_lost`] — and the census note is
//! made here, at the site that knows which rail it is. The set of sites and the
//! slug they record are unchanged.
//!
//! # The refusal vocabulary is mapped, not shared
//!
//! The rail crate declines with its own module-local
//! [`reims_vgpu_vulkan::pixel::Refusal`], because a crate that named this
//! device's [`TranslateReason`] would be a rail that depends on the device
//! model. [`reason`](super::reason) is still the one vocabulary this device
//! declines in, and the mapping below is where the two meet.

use ash::vk;

use reims_vgpu_vulkan::pixel as rail;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::{ColorAttachmentState, ColorClearValue, StorageImageFormat};
use crate::protocol::pixel_format::{self, SampledByteFormat, SwizzlePlan, TexelLayout};

pub use rail::{
    bytes_per_texel, has_bgra_order, has_identity_components, is_srgb, resident_color,
    sample_view_format, srgb_texel_layout, storage_format, storage_image_components,
    storage_image_from_selector, stored_bytes_agree, texel_layout_of, verbatim_texel,
    vk_block_geometry, vk_component_mapping, vk_storage_image, vk_texel_layout, PixelFormat,
    ResidentFormat, TransferFunction, RESIDENT_RGBA_FORMAT, SCANOUT_FORMAT, TRANSIENT_DEPTH_FORMAT,
};

/// This device's name for a decline the rail crate made in its own vocabulary.
///
/// Total, so a refusal the rail adds cannot arrive here as a default.
const fn declined(refusal: rail::Refusal) -> TranslateReason {
    match refusal {
        rail::Refusal::UnknownPixelFormat(mtl) => TranslateReason::UnknownPixelFormat(mtl),
        rail::Refusal::NoSampledLayout(mtl) => TranslateReason::NoSampledLayout(mtl),
        rail::Refusal::NoColorAttachmentFormat(mtl) => {
            TranslateReason::NoColorAttachmentFormat(mtl)
        }
        rail::Refusal::NoStorageImageFormat(mtl) => TranslateReason::NoStorageImageFormat(mtl),
    }
}

/// Translate one decoded `MTLPixelFormat`.
///
/// See [`reims_vgpu_vulkan::pixel::translate`] for what the table admits and
/// why it is total.
pub fn translate(mtl: u16) -> Result<PixelFormat, TranslateReason> {
    rail::translate(mtl).map_err(declined)
}

/// The guest texel layout for a decoded Metal pixel format, and the decline to
/// record if reaching it dropped the sRGB qualifier.
///
/// `Ok((layout, Some(reason), plan))` means the layout is right but the transfer
/// function was lost; `Ok((layout, None, plan))` means nothing was lost. The
/// table answers whether it was lost; this names the loss in the vocabulary the
/// rest of this crate declines in.
pub fn sampled_pixels(
    mtl: u16,
) -> Result<(TexelLayout, Option<TranslateReason>, SwizzlePlan), TranslateReason> {
    let sampled = rail::sampled_pixels(mtl).map_err(declined)?;
    let loss = sampled
        .srgb_lost
        .then_some(TranslateReason::SrgbDowngraded(mtl));
    Ok((sampled.layout, loss, sampled.components))
}

/// The engine's storage-image format for a Metal pixel format.
///
/// See [`reims_vgpu_vulkan::pixel::storage_image`] for why this rail keeps an
/// enum where the colour-attachment and sampled rails resolve to a `VkFormat`.
pub fn storage_image(mtl: u16) -> Result<StorageImageFormat, TranslateReason> {
    rail::storage_image(mtl).map_err(declined)
}

/// The compute path's admission for a **sampled** image bind.
///
/// A superset of [`storage_image`], for the reason
/// [`reims_vgpu_vulkan::pixel::sampled_image`] states: `SAMPLED_IMAGE` and
/// `STORAGE_IMAGE` are different Vulkan guarantees and the guest asks two
/// different questions.
pub fn sampled_image(mtl: u16) -> Result<StorageImageFormat, TranslateReason> {
    rail::sampled_image(mtl).map_err(declined)
}

/// Every Vulkan format a colour attachment may take, and the decline for a
/// format the rail does not render to.
///
/// The pair the table hands back is bound into a [`ColorAttachmentFormat`] here
/// so the format and the numeric type its clear value must use cannot be
/// carried apart — Vulkan keeps them in separate API objects and they are one
/// contract decision.
pub fn color_attachment(
    mtl: u16,
) -> Result<(ColorAttachmentFormat, Option<TranslateReason>), TranslateReason> {
    let (vk, numeric) = rail::color_attachment(mtl).map_err(declined)?;
    Ok((ColorAttachmentFormat { vk, numeric }, None))
}

/// A colour-renderable Metal format translated together with the numeric type
/// its clear value must use.
///
/// Vulkan keeps the image format and clear union member in separate API
/// objects, but they are one contract decision. Carrying them together keeps
/// an integer attachment from being created correctly and then cleared through
/// the float member later in command emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorAttachmentFormat {
    pub vk: vk::Format,
    numeric: pixel_format::ColorNumericType,
}

impl ColorAttachmentFormat {
    /// Lower Metal's double-precision clear carrier using this attachment's
    /// declared numeric interpretation.
    pub fn clear_value(self, components: [f64; 4]) -> ColorClearValue {
        ColorClearValue::from_components(self.numeric, components)
    }

    /// Bind this format to its clear value without exposing two independently
    /// mutable fields to the engine request.
    pub fn with_clear(self, components: [f64; 4]) -> ColorAttachmentState {
        ColorAttachmentState::new(self.vk, self.clear_value(components))
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

#[cfg(test)]
mod tests {

    /// Every Metal format a dispatch samples binds the component mapping
    /// `translate` states for it — not one the reduction to
    /// [`StorageImageFormat`] lost on the way.
    ///
    /// The reduction is where the plan goes missing. `A8Unorm` and `R8Unorm`
    /// are one `VkFormat`, and only the first needs its byte moved back into
    /// alpha, so an enum that named the format alone could not tell a rail which
    /// mapping to bind — and the compute rail refused `A8Unorm` outright for
    /// exactly that reason, taking ten conformance cases with it. This sweeps
    /// the whole 16-bit space rather than naming that one format, because the
    /// next format with a non-identity plan must not be able to slip through the
    /// same hole quietly.
    #[test]
    fn every_sampled_dispatch_format_binds_the_plan_its_metal_format_states() {
        for mtl in 0..=u16::MAX {
            let Ok(engine) = sampled_image(mtl) else {
                continue;
            };
            let stated = translate(mtl)
                .expect("a format sampled_image admitted must translate")
                .components;
            assert_eq!(
                storage_image_components(engine),
                stated,
                "{mtl:#x} samples as {engine:?} but would bind the wrong channels"
            );
        }
    }

    /// `A8Unorm` reaches a dispatch, and it reaches it as alpha.
    ///
    /// The named case of the sweep above, spelled out because the failure it
    /// guards is silent: sampling the byte as **red** returns a plausible
    /// non-zero value in the wrong channel, which reads like a working texture.
    #[test]
    fn a8unorm_samples_in_a_dispatch_with_its_byte_in_alpha() {
        use crate::protocol::pixel_format as p;
        let engine =
            sampled_image(p::MTL_FORMAT_A8_UNORM).expect("A8Unorm is sampleable in a dispatch");
        assert_ne!(
            engine,
            StorageImageFormat::R8Unorm,
            "A8Unorm must not reduce to R8Unorm: they share a VkFormat and \
             differ only in the mapping, so collapsing them loses the mapping"
        );
        assert_eq!(vk_storage_image(engine), vk::Format::R8_UNORM);
        assert_eq!(engine.bytes_per_texel(), 1);
        let mapping = vk_component_mapping(&storage_image_components(engine));
        assert_eq!(
            (mapping.r, mapping.g, mapping.b, mapping.a),
            (
                vk::ComponentSwizzle::ZERO,
                vk::ComponentSwizzle::ZERO,
                vk::ComponentSwizzle::ZERO,
                vk::ComponentSwizzle::R,
            ),
            "Metal A8Unorm presents (0, 0, 0, a)"
        );
        assert_eq!(
            storage_image_components(StorageImageFormat::R8Unorm),
            p::swizzle_identity(),
            "its VkFormat twin keeps the identity mapping"
        );
    }
    use super::*;
    use crate::observe::Decline;
    use pixel_format as p;

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
                format.vk,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            ));
            assert_eq!(decline, None, "colour attachment must preserve sRGB");
        }
        // The converse: a linear format must never produce the decline, or the
        // proxy floods and stops meaning anything.
        // Swept over the whole domain rather than over a list: the list of
        // formats the table accepts is the table's own, and it lives with the
        // table now.
        for mtl in 0..=u16::MAX {
            if is_srgb(mtl) {
                continue;
            }
            if let Ok((_, decline, _)) = sampled_pixels(mtl) {
                assert_eq!(decline, None, "MTL {mtl:#x}");
            }
            if let Ok((_, decline)) = color_attachment(mtl) {
                assert_eq!(decline, None, "MTL {mtl:#x}");
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
            (p::MTL_FORMAT_RGBA8_UNORM_SRGB, vk::Format::R8G8B8A8_SRGB),
            (p::MTL_FORMAT_BGRA8_UNORM_SRGB, vk::Format::B8G8R8A8_SRGB),
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
            let allocation = ResidentFormat::of(attachment.vk).allocation();
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
            color_attachment(p::MTL_FORMAT_BGRA8_UNORM_SRGB)
                .unwrap()
                .0
                .vk,
            vk::Format::B8G8R8A8_SRGB
        );
        assert_eq!(
            color_attachment(p::MTL_FORMAT_RGBA8_UNORM_SRGB)
                .unwrap()
                .0
                .vk,
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

    /// A linear `RG16Uint` guest attachment is served end to end, and never
    /// through an eight-bit intermediate.
    ///
    /// The macos-15 defect. The format was absent from this crate entirely — no
    /// constant, no width, no layout — so a guest that rendered into one lost
    /// the whole pass and a guest that sampled one was refused as
    /// `unknown_pixel_format`.
    ///
    /// What makes it renderable is not an entry in a table: it is that both
    /// Store rails can land its texel without inventing a byte. The GPU-direct
    /// arm always could, given `store_texel_order`; the copying arm learned to,
    /// once a readback carried [`ReadbackTexel`] instead of a bare "is it
    /// BGRA" flag. The eight-bit rails must stay shut, and that is asserted
    /// here rather than described, because an arm added to one of them later
    /// would silently start quantizing a count into a fraction.
    #[test]
    fn an_integer_colour_attachment_is_served_by_the_native_rail_only() {
        let (attachment, decline) = color_attachment(p::MTL_FORMAT_RG16_UINT).unwrap();
        let format = attachment.vk;
        assert_eq!(format, vk::Format::R16G16_UINT);
        assert_eq!(decline, None);
        assert_eq!(
            attachment.clear_value([1.0, 2.0, 65_535.0, 0.0]),
            ColorClearValue::Uint([1, 2, 65_535, 0]),
            "the clear is a numeric conversion, not the float bit pattern"
        );
        assert_eq!(bytes_per_texel(format), Some(p::RG16_BPP));
        // Its own Vulkan format, distinct from the sibling it shares bytes
        // with: one format for both would read every texel as a fraction of
        // full scale and paint a wrong frame rather than refuse one.
        assert_ne!(format, translate(p::MTL_FORMAT_RG16_UNORM).unwrap().vk);
        assert_eq!(texel_layout_of(format), Some(TexelLayout::Rg16Uint));
        assert_eq!(vk_texel_layout(TexelLayout::Rg16Uint), format);
        // Sampled and dispatch rails carry it too.
        assert!(sampled_pixels(p::MTL_FORMAT_RG16_UINT).is_ok());
        assert!(sampled_image(p::MTL_FORMAT_RG16_UINT).is_ok());
        // The native Store rail is the one that lands it, on both arms.
        assert_eq!(
            p::store_texel_order(p::MTL_FORMAT_RG16_UINT),
            Some(TexelLayout::Rg16Uint)
        );
        // And every eight-bit rail stays shut, in both directions.
        const PX: u32 = 4;
        let wide = vec![0u8; PX as usize * p::RG16_BPP as usize];
        let mut rgba = vec![0u8; PX as usize * p::RGBA8_BPP as usize];
        assert!(!p::narrow_texel_to_rgba8(
            TexelLayout::Rg16Uint,
            &wide,
            PX,
            &mut rgba
        ));
        let mut back = wide.clone();
        assert!(!p::expand_rgba8_to_texel(
            TexelLayout::Rg16Uint,
            &rgba,
            PX,
            &mut back
        ));
        assert!(!p::convert_rgba8_to_row(
            p::MTL_FORMAT_RG16_UINT,
            &rgba,
            PX,
            &mut back
        ));
        assert!(!p::solid_color_reaches_texel(p::MTL_FORMAT_RG16_UINT));
    }

    #[test]
    fn continuous_colour_attachments_keep_semantic_float_clears() {
        let attachment = color_attachment(p::MTL_FORMAT_BGRA8_UNORM_SRGB).unwrap().0;
        assert_eq!(
            attachment.clear_value([0.25, 0.5, 0.75, 1.0]),
            ColorClearValue::Float([0.25, 0.5, 0.75, 1.0])
        );
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
            crate::protocol::pixel_format::RGBA16F_BPP
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
        assert_eq!(a8_plan, rail::ALPHA_IN_RED);
        assert_eq!(
            sampled_pixels(p::MTL_FORMAT_R8_UNORM).unwrap().2,
            rail::IDENTITY,
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
        use crate::protocol::pixel_format::TexelLayout;
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
        use crate::protocol::pixel_format::TexelLayout;
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

    /// Every rail's accepted set, spelled out. A format silently joining or
    /// leaving one of these changes which draws take the zero-copy path.
    /// Every `MTLPixelFormat` the table accepts, ascending.
    ///
    /// The list itself belongs to the table and lives with it — see
    /// `reims_vgpu_vulkan::pixel`'s `expected_names_every_format_the_table_
    /// translates`, which holds the literal there equal to this sweep. Derived
    /// here rather than spelled a second time, so a format added to the table
    /// reaches these rail-membership tests without a second edit that a commit
    /// could half-land.
    fn translated() -> impl Iterator<Item = u16> {
        (0..=u16::MAX).filter(|mtl| translate(*mtl).is_ok())
    }

    /// A membership answer as a set the order of the sweep cannot change.
    fn ascending(formats: impl IntoIterator<Item = u16>) -> Vec<u16> {
        let mut v: Vec<u16> = formats.into_iter().collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn the_engine_rails_accept_exactly_these_formats() {
        let sampled = ascending(translated().filter(|&mtl| sampled_pixels(mtl).is_ok()));
        assert_eq!(
            sampled,
            ascending(vec![
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
                // Its integer sibling, sampled on the same terms. It is *not*
                // in the colour list below, which is the distinction this
                // format exists in the tables to make: known and sampleable,
                // and still not something this device renders into.
                p::MTL_FORMAT_RG16_UINT,
                p::MTL_FORMAT_RGBA16_UNORM,
                p::MTL_FORMAT_RGBA16_FLOAT,
                // Four-channel `float32`. A macos-15 guest binds 1x1 and 4x1
                // linear textures of it to a **vertex** sampler and every draw
                // that did was refused, because this crate had no layout for the
                // format at all. It is sampled-and-dispatched only: it never
                // appears in the colour list below, and it has no CPU narrowing
                // arm by design — see `TexelLayout::Rgba32Float`.
                p::MTL_FORMAT_RGBA32_FLOAT,
            ])
        );
        let color = ascending(translated().filter(|&mtl| color_attachment(mtl).is_ok()));
        assert_eq!(
            color,
            ascending(vec![
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
                // `draw::render_target`'s `rt_backing_base_format` until the
                // FourCC and `render_target_bpp` both named it.
                //
                // Unlike every other member here this one is not a format
                // Vulkan mandates for `COLOR_ATTACHMENT_BIT`, so a host may
                // decline it and this rail is where that decline appears. The
                // NVIDIA host it was measured on advertises it.
                p::MTL_FORMAT_BGR10A2_UNORM,
                // Its red-low sibling, which macOS 26's icon renderer draws
                // into as linear GVA targets — every one refused as
                // `rt_resolve reason=rt_linear_format fmt=0x5a`, and every app
                // icon blank. Its Vulkan twin `A2B10G10R10_UNORM_PACK32`, unlike
                // the member above, *is* mandated for colour attachment and
                // blend, so no host can decline it.
                p::MTL_FORMAT_RGB10A2_UNORM,
                // The first **integer** colour attachment, and the one that
                // could not be admitted by adding a table entry. A macos-15
                // guest renders into linear `RG16Uint` targets, and every pass
                // naming one was refused as `rt_resolve reason=rt_linear_format`
                // — seven dropped clears and thirty-two unresolved MRT slots in
                // one boot.
                //
                // The three eight-bit conversion arms the members above needed
                // are the arms this format must never have: an integer texel is
                // a count, so there is no unorm8 byte that stands for it. It is
                // renderable because the *native* rail exists instead — the
                // GPU-direct copy and, since the readback learned to carry its
                // own texel, the copying arm as well.
                p::MTL_FORMAT_RG16_UINT,
                // Admitted since the two arms became one answer. The contract
                // has said a half-float render target is renderable since one
                // could be created at that format; only this side still refused
                // it, so a half-float secondary MRT slot was declined while a
                // half-float primary was not.
                p::MTL_FORMAT_RGBA16_FLOAT,
            ])
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
        for mtl in translated() {
            let admitted = color_attachment(mtl).is_ok();
            assert_eq!(
                admitted,
                p::render_target_bpp(mtl).is_some(),
                "{mtl:#x}: the two colour-attachment arms disagree"
            );
            if !admitted {
                continue;
            }
            let format = color_attachment(mtl).unwrap().0.vk;
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
            // A format whose texel has no eight-bit form owes none of the three
            // eight-bit rails — it owes the **native** one instead, which is
            // `store_texel_order` naming this same layout. That is a whole rail
            // and not an exemption: the GPU-direct arm copies the resident into
            // the guest's pages, and the copying arm lands the readback's own
            // texel verbatim through `FrameRows::Native`, so both routes serve
            // it exactly and neither invents a byte.
            //
            // Asserting the alternative rather than skipping is the point. A
            // renderable format that satisfies neither set is the silent
            // frame-loss this test was written to catch.
            if !p::narrow_texel_to_rgba8(layout, &wide, PX, &mut rgba) {
                assert_eq!(
                    p::store_texel_order(mtl),
                    Some(layout),
                    "{mtl:#x}: renderable as {layout:?}, which neither narrows to \
                     RGBA8 nor is a native byte-copy destination — so no rail lands it"
                );
                assert!(
                    !p::solid_color_reaches_texel(mtl),
                    "{mtl:#x}: has no readback narrowing but does take a solid \
                     colour, so the two eight-bit tables disagree about it"
                );
                continue;
            }
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
        for mtl in translated() {
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
            saw_non_identity |= plan != rail::IDENTITY;
            assert_eq!(has_identity_components(mtl), plan == rail::IDENTITY);
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
