//! Device-capability half of the translation boundary: given the format a
//! Metal value *means*, what can the bound GPU actually do with it?
//!
//! Translation answers meaning; this answers capability. Splitting them is what
//! keeps "which host GPU is in the machine" from leaking into "what does this
//! Metal value mean" — the mistake the four-cell matrix exists to prevent.
//!
//! # Vertex buffer formats are not all mandatory
//!
//! Vulkan requires only a subset of formats to be usable as vertex attributes.
//! **Every three-component 8- and 16-bit format is outside that subset** —
//! `R8G8B8_*` and `R16G16B16_*` are optional, while their four-component
//! siblings and all the `R32G32B32_*` formats are mandatory. A guest that binds
//! a `MTLVertexFormatShort3` attribute on a device that declines
//! `R16G16B16_UINT` used to lose the whole draw.
//!
//! # The widening fallback
//!
//! A declined three-component format is substituted by its four-component
//! sibling, which is mandatory everywhere. The first three components are exact:
//! these formats are component-packed, so components 0..2 sit at identical byte
//! offsets in both, and the three values the shader reads are the same bytes.
//!
//! **The fourth component is not, and that is the whole of the difficulty.**
//! Vulkan supplies a vertex input's missing components from the constant vector
//! `(0, 0, 0, 1)`, so a shader input declared `vec4` over the guest's own
//! three-component format takes `(x, y, z, 1.0)`. Over the four-component
//! substitute it takes `(x, y, z, <whatever those bytes hold>)` — the next
//! attribute's data, or the vertex's padding — because the component is now
//! *supplied* rather than defaulted. Nothing about the bytes is wrong; what
//! changed is that Vulkan stopped filling in a default the guest was relying on.
//!
//! This module used to assert the substitution was exact and give as its second
//! reason that "the shader's input variable is a three-component vector". That
//! is the condition, not a fact — nothing checked it, and a `vec4` reader is
//! exactly the shape that makes it false. So `VertexFormatSupport::resolve` now asks
//! [`crate::spirv_vertex_input`] what the shader declares at the
//! attribute's location, and widens only where the answer makes the
//! substitution invisible:
//!
//! | shader declares | verdict |
//! |---|---|
//! | 3 components or fewer | widen — Vulkan discards what the format oversupplies |
//! | 4 components | refuse `vertex_format_widen_read_as_four` |
//! | nothing at this location | widen — an input the shader never reads |
//! | a type the walk cannot measure | refuse `vertex_format_widen_shader_unreadable` |
//!
//! Refusing rather than widening anyway is the choice `AGENTS.md` asks for: a
//! GPU refuses a request it cannot represent, and a wrong `w` reaching a vertex
//! shader is a geometry error with nothing downstream able to name it.
//!
//! Widening is also only safe when the wider read stays inside the vertex
//! buffer. The widened attribute reads `bytes_wide` at `offset` within each
//! vertex, so the last vertex's read runs past the buffer unless
//! `offset + bytes_wide <= stride`. That condition is checked, and a substitution
//! that would not fit declines by name instead — a read past the end of a
//! vertex buffer is a Vulkan violation, not a degraded frame.
//!
//! # The cost is paid only where the fallback is reached
//!
//! `unsupported` is empty on every host this project has run on, so `resolve`
//! returns the attribute's own format before it ever looks at a shader. The
//! width is therefore passed as a closure rather than a value: a host that
//! widens nothing never walks a SPIR-V module, and a host that widens walks it
//! once per pipeline miss no matter how many attributes need it.

use ash::vk;

use super::reason::TranslateReason;
use super::vertex::{self, VertexLayout};
use crate::engine::VertexAttributeFormat;
use crate::spirv_vertex_input::InputWidth;

/// What the pipeline should bind for one vertex attribute on this device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBinding {
    pub format: vk::Format,
    /// `Some(original)` when the device declined the attribute's own format and
    /// the mandatory wider sibling was substituted. Callers report this so a
    /// widened pipeline is visible rather than assumed.
    pub widened_from: Option<vk::Format>,
}

/// Which vertex attribute formats the bound device accepts, resolved once.
///
/// The set this backend can emit is small and fixed, so the whole thing is
/// probed at device create rather than queried per pipeline — the same shape as
/// classifying memory properties once instead of re-querying per allocation.
#[derive(Clone, Debug, Default)]
pub struct VertexFormatSupport {
    /// Raw `VkFormat` values the device does NOT accept as a vertex buffer
    /// format. Stored as the negative set because it is empty on every device
    /// seen so far, and an empty `Vec` makes the common lookup a length check.
    unsupported: Vec<i32>,
}

impl VertexFormatSupport {
    /// Probe every format this backend can emit for a vertex attribute.
    ///
    /// # Safety
    ///
    /// `instance` and `pd` must be live. The call itself only reads properties.
    pub fn probe(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Self {
        let mut unsupported = Vec::new();
        for format in emittable_formats() {
            // SAFETY: reading physical-device format properties requires only a
            // live instance and physical device, both of which the caller owns.
            let props = unsafe { instance.get_physical_device_format_properties(pd, format) };
            if !props
                .buffer_features
                .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
            {
                unsupported.push(format.as_raw());
            }
        }
        unsupported.sort_unstable();
        Self { unsupported }
    }

    /// Build from an explicit unsupported set. Lets the fallback be tested on
    /// every matrix row without owning that row's hardware.
    pub fn with_unsupported(formats: &[vk::Format]) -> Self {
        let mut unsupported: Vec<i32> = formats.iter().map(|f| f.as_raw()).collect();
        unsupported.sort_unstable();
        Self { unsupported }
    }

    pub fn accepts(&self, format: vk::Format) -> bool {
        self.unsupported.binary_search(&format.as_raw()).is_err()
    }

    /// Resolve one attribute against this device.
    ///
    /// `offset` and `stride` are the attribute's own placement in the vertex
    /// buffer and decide whether the widening fallback is in bounds.
    ///
    /// `shader_width` is called at most once, and only on the path that would
    /// substitute a wider format — see this module's doc for why it is a
    /// closure. It must answer for *this attribute's* `Location`.
    pub fn resolve(
        &self,
        format: VertexAttributeFormat,
        offset: u32,
        stride: u32,
        shader_width: impl FnOnce() -> InputWidth,
    ) -> Result<VertexBinding, TranslateReason> {
        let layout = vertex::vertex_layout(format);
        if self.accepts(layout.vk) {
            return Ok(VertexBinding {
                format: layout.vk,
                widened_from: None,
            });
        }
        let Some(wide) = widened(&layout) else {
            return Err(TranslateReason::FormatNotVertexBuffer(layout.vk.as_raw()));
        };
        // The substitute must itself be accepted, and the wider read must stay
        // inside the vertex. A stride of 0 means "one element, tightly packed"
        // in Metal's stage-in encoding and leaves no room to widen into.
        let fits = stride > 0 && offset.saturating_add(wide.bytes) <= stride;
        if !self.accepts(wide.vk) || !fits {
            return Err(TranslateReason::FormatNotVertexBuffer(layout.vk.as_raw()));
        }
        // Asked last, so the two cheap structural questions above answer first
        // and a module is walked only for an attribute that would really be
        // substituted.
        match widening_is_invisible(shader_width()) {
            Ok(()) => Ok(VertexBinding {
                format: wide.vk,
                widened_from: Some(layout.vk),
            }),
            Err(reason) => Err(reason(layout.vk.as_raw())),
        }
    }
}

/// Whether substituting a four-component format changes what the shader reads.
///
/// One home for the rule the module doc tabulates, so a second caller cannot
/// write a fourth version of it. Returns the constructor of the refusal rather
/// than the refusal itself because the payload — the format the guest asked for
/// — belongs to the caller.
fn widening_is_invisible(width: InputWidth) -> Result<(), fn(i32) -> TranslateReason> {
    match width {
        // Vulkan discards components an attribute format supplies past the
        // shader's declared width, so the oversupplied fourth is never read.
        InputWidth::Components(n) if n <= 3 => Ok(()),
        // The shader reads a component the guest's own format would have had
        // Vulkan default to 1.0.
        InputWidth::Components(_) => Err(TranslateReason::VertexFormatWidenReadAsFour),
        // The vertex descriptor describes an attribute this shader does not
        // declare. Nothing reads the substitute, so nothing can tell.
        InputWidth::Absent => Ok(()),
        InputWidth::Unreadable => Err(TranslateReason::VertexFormatWidenShaderUnreadable),
    }
}

/// The mandatory four-component sibling of a three-component vertex format.
///
/// `None` for anything else: two- and four-component formats are already
/// mandatory, and the packed 32-bit formats have no wider sibling that keeps
/// their bit layout.
fn widened(layout: &VertexLayout) -> Option<VertexLayout> {
    if layout.components != 3 {
        return None;
    }
    let (vk, bytes) = match layout.vk {
        vk::Format::R8G8B8_UINT => (vk::Format::R8G8B8A8_UINT, 4),
        vk::Format::R8G8B8_UNORM => (vk::Format::R8G8B8A8_UNORM, 4),
        vk::Format::R8G8B8_SNORM => (vk::Format::R8G8B8A8_SNORM, 4),
        vk::Format::R16G16B16_UINT => (vk::Format::R16G16B16A16_UINT, 8),
        vk::Format::R16G16B16_UNORM => (vk::Format::R16G16B16A16_UNORM, 8),
        vk::Format::R16G16B16_SNORM => (vk::Format::R16G16B16A16_SNORM, 8),
        vk::Format::R16G16B16_SFLOAT => (vk::Format::R16G16B16A16_SFLOAT, 8),
        // R32G32B32_* are already mandatory, and the packed 32-bit
        // three-component formats (B10G11R11, E5B9G9R9) occupy one word with no
        // wider equivalent — widening either would change the bit layout.
        _ => return None,
    };
    Some(VertexLayout {
        vk,
        bytes,
        components: 4,
    })
}

/// Every `VkFormat` a vertex attribute can resolve to, derived from the
/// attribute table so the probe cannot fall behind it.
fn emittable_formats() -> Vec<vk::Format> {
    let mut out: Vec<vk::Format> = Vec::new();
    let mut push = |f: vk::Format| {
        if !out.contains(&f) {
            out.push(f);
        }
    };
    for format in ALL_ATTRIBUTE_FORMATS.iter() {
        let layout = vertex::vertex_layout(*format);
        push(layout.vk);
        if let Some(wide) = widened(&layout) {
            push(wide.vk);
        }
    }
    out
}

/// Every attribute format, reachable from the wire decode. Derived by walking
/// the `MTLVertexFormat` range rather than restating the enum, so a new format
/// joins the probe automatically.
static ALL_ATTRIBUTE_FORMATS: std::sync::LazyLock<Vec<VertexAttributeFormat>> =
    std::sync::LazyLock::new(|| {
        (0..=64u32)
            .filter_map(|m| vertex::attribute_format(m).ok())
            .collect()
    });

#[cfg(test)]
mod tests {
    use super::*;

    /// A shader that reads three components — the width that licenses the
    /// substitution — for the tests below whose subject is the capability
    /// question rather than the shader question.
    fn three() -> InputWidth {
        InputWidth::Components(3)
    }

    /// The probe covers every format the attribute table can produce plus every
    /// widening substitute — a format missing from the probe would be assumed
    /// supported and fail at pipeline create instead of declining by name.
    #[test]
    fn the_probe_covers_every_emittable_format() {
        let probed = emittable_formats();
        for format in ALL_ATTRIBUTE_FORMATS.iter() {
            let layout = vertex::vertex_layout(*format);
            assert!(probed.contains(&layout.vk), "{format:?} missing from probe");
            if let Some(wide) = widened(&layout) {
                assert!(
                    probed.contains(&wide.vk),
                    "{format:?} widening target missing from probe"
                );
            }
        }
        // 53 attribute formats collapse onto far fewer distinct Vulkan formats.
        assert!(probed.len() >= 30, "probed {}", probed.len());
        assert_eq!(ALL_ATTRIBUTE_FORMATS.len(), 53);
    }

    /// On a device that accepts everything — every host this project has run on
    /// — nothing is widened and the resolved format is the attribute's own.
    #[test]
    fn a_permissive_device_widens_nothing() {
        let support = VertexFormatSupport::default();
        for format in ALL_ATTRIBUTE_FORMATS.iter() {
            let binding = support.resolve(*format, 0, 64, three).unwrap();
            assert_eq!(binding.format, vertex::vk_format(*format), "{format:?}");
            assert_eq!(binding.widened_from, None, "{format:?}");
        }
    }

    /// The nine three-component 8/16-bit attribute formats are exactly the ones
    /// Vulkan does not require, and exactly the ones that can widen.
    #[test]
    fn the_nine_optional_three_component_formats_can_widen() {
        use VertexAttributeFormat as F;
        let optional = [
            F::UChar3,
            F::Char3,
            F::UChar3Normalized,
            F::Char3Normalized,
            F::UShort3,
            F::Short3,
            F::UShort3Normalized,
            F::Short3Normalized,
            F::Half3,
        ];
        assert_eq!(optional.len(), 9);
        let widenable: Vec<F> = ALL_ATTRIBUTE_FORMATS
            .iter()
            .copied()
            .filter(|f| widened(&vertex::vertex_layout(*f)).is_some())
            .collect();
        assert_eq!(widenable, optional);
        // The 32-bit three-component formats are mandatory, so they do not and
        // need not widen.
        for f in [F::Float3, F::Int3, F::UInt3] {
            assert!(widened(&vertex::vertex_layout(f)).is_none(), "{f:?}");
        }
    }

    /// A device that declines the optional format binds the mandatory sibling
    /// instead of losing the draw — and says that it did.
    #[test]
    fn a_declined_three_component_format_widens_to_its_mandatory_sibling() {
        let support = VertexFormatSupport::with_unsupported(&[
            vk::Format::R16G16B16_SFLOAT,
            vk::Format::R8G8B8_UNORM,
        ]);
        // Half3 is 6 bytes; the substitute is 8 and fits in a 16-byte stride.
        let binding = support
            .resolve(VertexAttributeFormat::Half3, 0, 16, three)
            .unwrap();
        assert_eq!(binding.format, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(binding.widened_from, Some(vk::Format::R16G16B16_SFLOAT));

        let binding = support
            .resolve(VertexAttributeFormat::UChar3Normalized, 4, 12, three)
            .unwrap();
        assert_eq!(binding.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(binding.widened_from, Some(vk::Format::R8G8B8_UNORM));

        // An untouched format on the same device still resolves natively.
        let binding = support
            .resolve(VertexAttributeFormat::Float4, 0, 16, three)
            .unwrap();
        assert_eq!(binding.format, vk::Format::R32G32B32A32_SFLOAT);
        assert_eq!(binding.widened_from, None);
    }

    /// The fallback is refused when the wider read would run past the vertex —
    /// reading off the end of a vertex buffer is a spec violation, not a
    /// degraded frame, so declining by name is the only correct answer.
    #[test]
    fn widening_is_refused_when_it_would_read_past_the_vertex() {
        let support = VertexFormatSupport::with_unsupported(&[vk::Format::R16G16B16_SFLOAT]);
        // Stride exactly the narrow size: no room for the 8-byte substitute.
        assert_eq!(
            support
                .resolve(VertexAttributeFormat::Half3, 0, 6, three)
                .unwrap_err(),
            TranslateReason::FormatNotVertexBuffer(vk::Format::R16G16B16_SFLOAT.as_raw())
        );
        // Fits by width but the attribute sits too late in the vertex.
        assert_eq!(
            support
                .resolve(VertexAttributeFormat::Half3, 10, 16, three)
                .unwrap_err(),
            TranslateReason::FormatNotVertexBuffer(vk::Format::R16G16B16_SFLOAT.as_raw())
        );
        // Stride 0 (single tightly-packed element) leaves nothing to widen into.
        assert_eq!(
            support
                .resolve(VertexAttributeFormat::Half3, 0, 0, three)
                .unwrap_err(),
            TranslateReason::FormatNotVertexBuffer(vk::Format::R16G16B16_SFLOAT.as_raw())
        );
        // Exactly enough room is enough.
        assert!(support
            .resolve(VertexAttributeFormat::Half3, 0, 8, three)
            .is_ok());
    }

    /// A declined format with no wider sibling declines by name rather than
    /// falling back to something with a different bit layout.
    #[test]
    fn a_format_with_no_sibling_declines_instead_of_guessing() {
        let support = VertexFormatSupport::with_unsupported(&[
            vk::Format::A2B10G10R10_SNORM_PACK32,
            vk::Format::R16G16B16A16_SFLOAT,
        ]);
        assert_eq!(
            support
                .resolve(VertexAttributeFormat::Int1010102Normalized, 0, 32, three)
                .unwrap_err(),
            TranslateReason::FormatNotVertexBuffer(vk::Format::A2B10G10R10_SNORM_PACK32.as_raw())
        );
        // The substitute itself being declined is also a decline, not a third
        // guess.
        let support = VertexFormatSupport::with_unsupported(&[
            vk::Format::R16G16B16_SFLOAT,
            vk::Format::R16G16B16A16_SFLOAT,
        ]);
        assert_eq!(
            support
                .resolve(VertexAttributeFormat::Half3, 0, 32, three)
                .unwrap_err(),
            TranslateReason::FormatNotVertexBuffer(vk::Format::R16G16B16_SFLOAT.as_raw())
        );
    }

    /// The reason this module stopped calling the substitution exact. A shader
    /// reading four components takes the fourth from the vertex buffer under
    /// the substitute and from Vulkan's `(0, 0, 0, 1)` default under the format
    /// the guest asked for, so the two are different geometry and the pipeline
    /// is refused rather than built.
    #[test]
    fn a_shader_reading_four_components_refuses_the_substitution() {
        let support = VertexFormatSupport::with_unsupported(&[vk::Format::R16G16B16_SFLOAT]);
        assert_eq!(
            support
                .resolve(VertexAttributeFormat::Half3, 0, 16, || {
                    InputWidth::Components(4)
                })
                .unwrap_err(),
            TranslateReason::VertexFormatWidenReadAsFour(vk::Format::R16G16B16_SFLOAT.as_raw())
        );
    }

    /// Every width the substitution *is* invisible to still widens: Vulkan
    /// discards components the format oversupplies past the shader's declared
    /// width, and an attribute the shader does not declare is read by nothing.
    #[test]
    fn every_width_the_shader_cannot_observe_still_widens() {
        let support = VertexFormatSupport::with_unsupported(&[vk::Format::R16G16B16_SFLOAT]);
        for width in [
            InputWidth::Components(1),
            InputWidth::Components(2),
            InputWidth::Components(3),
            InputWidth::Absent,
        ] {
            let binding = support
                .resolve(VertexAttributeFormat::Half3, 0, 16, || width)
                .unwrap_or_else(|e| panic!("{width:?} refused with {e}"));
            assert_eq!(binding.format, vk::Format::R16G16B16A16_SFLOAT, "{width:?}");
            assert_eq!(
                binding.widened_from,
                Some(vk::Format::R16G16B16_SFLOAT),
                "{width:?}"
            );
        }
    }

    /// A width the walk could not measure refuses, and under its own slug: the
    /// permissive reading would widen under a shader that might read the fourth
    /// component, and the repair for this one is to teach the walk the shape it
    /// met rather than to accept a wrong `w`.
    #[test]
    fn an_unreadable_shader_refuses_under_its_own_name() {
        let support = VertexFormatSupport::with_unsupported(&[vk::Format::R8G8B8_UNORM]);
        let err = support
            .resolve(VertexAttributeFormat::UChar3Normalized, 0, 8, || {
                InputWidth::Unreadable
            })
            .unwrap_err();
        assert_eq!(
            err,
            TranslateReason::VertexFormatWidenShaderUnreadable(vk::Format::R8G8B8_UNORM.as_raw())
        );
        assert_ne!(
            reims_vgpu_observe::Decline::slug(&err),
            reims_vgpu_observe::Decline::slug(&TranslateReason::VertexFormatWidenReadAsFour(0))
        );
    }

    /// The shader is asked only where the answer can change the outcome. A
    /// permissive device resolves every format natively, and a device that
    /// declines a format with no substitute refuses before asking — so the
    /// SPIR-V walk is never paid for on a host that widens nothing.
    #[test]
    fn the_shader_is_not_consulted_unless_a_substitution_is_on_the_table() {
        let asked = std::cell::Cell::new(0u32);
        let ask = || {
            asked.set(asked.get() + 1);
            InputWidth::Components(3)
        };

        let permissive = VertexFormatSupport::default();
        for format in ALL_ATTRIBUTE_FORMATS.iter() {
            permissive.resolve(*format, 0, 64, ask).unwrap();
        }
        assert_eq!(asked.get(), 0, "a permissive device consulted the shader");

        // Declined with no wider sibling: refused before the shader is asked.
        let no_sibling =
            VertexFormatSupport::with_unsupported(&[vk::Format::A2B10G10R10_SNORM_PACK32]);
        no_sibling
            .resolve(VertexAttributeFormat::Int1010102Normalized, 0, 32, ask)
            .unwrap_err();
        assert_eq!(asked.get(), 0);

        // Declined and the substitute would not fit: also refused before.
        let too_tight = VertexFormatSupport::with_unsupported(&[vk::Format::R16G16B16_SFLOAT]);
        too_tight
            .resolve(VertexAttributeFormat::Half3, 0, 6, ask)
            .unwrap_err();
        assert_eq!(asked.get(), 0);

        // Declined, substitutable and in bounds — now the answer matters.
        too_tight
            .resolve(VertexAttributeFormat::Half3, 0, 16, ask)
            .unwrap();
        assert_eq!(asked.get(), 1);
    }
}
