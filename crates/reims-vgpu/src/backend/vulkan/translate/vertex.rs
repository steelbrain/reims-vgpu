//! `MTLVertexFormat` → the engine's attribute format → `VkFormat`, with the
//! attribute's byte size in the same table.
//!
//! # Why one table
//!
//! The Vulkan spelling and the byte size are the same fact stated two ways: a
//! `Short3` occupies six bytes *because* it is three 16-bit components, which is
//! also why it is `R16G16B16_UINT`. Held in two separate matches they drift —
//! one arm gets fixed, the other keeps the old answer, and the mismatch surfaces
//! as a stride bug in a shader nobody is looking at. [`vertex_layout`] states
//! both once.
//!
//! # The signedness ABI
//!
//! Twelve signed Metal formats map to **unsigned** Vulkan formats. That is
//! correct and it is not a guess: metal2vulkan's native emitter mints every
//! LLVM integer as `OpTypeInt <width> 0` (LLVM integers carry no signedness for
//! it to preserve), and its stage-input pass declares a vertex attribute's Input
//! variable with the AIR parameter's own type verbatim. So every integer vertex
//! stage input in emitted SPIR-V is unsigned, and Vulkan's numeric-type match
//! rule makes a `*_UINT` attribute format the only conforming pairing — a
//! `*_SINT` format against these modules is the undefined case. For the 32-bit
//! arms the pairing is bit-exact: nothing is extended, and the shader body's
//! signed instructions read the intended value.
//!
//! This is a **cross-repository ABI coupling**. If metal2vulkan ever emits a
//! genuinely signed stage input, every signed arm below becomes wrong, silently.
//! [`SIGNED_AS_UNSIGNED`] names the arms that depend on it so the blast radius
//! is enumerable rather than a comment on one line of twelve.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::{VertexAttributeFormat as F, VertexStepFunction};

/// One attribute format's Vulkan spelling and its footprint in the vertex
/// buffer, stated together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexLayout {
    pub vk: vk::Format,
    /// Bytes this attribute occupies in the guest's vertex buffer.
    pub bytes: u32,
    /// Component count, which decides whether a widening fallback exists when
    /// the device declines the format (see [`super::support`]).
    pub components: u32,
}

const fn layout(vk: vk::Format, bytes: u32, components: u32) -> VertexLayout {
    VertexLayout {
        vk,
        bytes,
        components,
    }
}

/// The single table. Every arm states the Vulkan format, the byte size and the
/// component count together, so no two can disagree.
pub fn vertex_layout(format: F) -> VertexLayout {
    match format {
        F::UChar => layout(vk::Format::R8_UINT, 1, 1),
        F::Char => layout(vk::Format::R8_UINT, 1, 1),
        F::UCharNormalized => layout(vk::Format::R8_UNORM, 1, 1),
        F::CharNormalized => layout(vk::Format::R8_SNORM, 1, 1),

        F::UChar2 => layout(vk::Format::R8G8_UINT, 2, 2),
        F::Char2 => layout(vk::Format::R8G8_UINT, 2, 2),
        F::UChar2Normalized => layout(vk::Format::R8G8_UNORM, 2, 2),
        F::Char2Normalized => layout(vk::Format::R8G8_SNORM, 2, 2),

        F::UChar3 => layout(vk::Format::R8G8B8_UINT, 3, 3),
        F::Char3 => layout(vk::Format::R8G8B8_UINT, 3, 3),
        F::UChar3Normalized => layout(vk::Format::R8G8B8_UNORM, 3, 3),
        F::Char3Normalized => layout(vk::Format::R8G8B8_SNORM, 3, 3),

        F::UChar4 => layout(vk::Format::R8G8B8A8_UINT, 4, 4),
        F::Char4 => layout(vk::Format::R8G8B8A8_UINT, 4, 4),
        F::UChar4Normalized => layout(vk::Format::R8G8B8A8_UNORM, 4, 4),
        F::Char4Normalized => layout(vk::Format::R8G8B8A8_SNORM, 4, 4),
        F::UChar4NormalizedBgra => layout(vk::Format::B8G8R8A8_UNORM, 4, 4),

        F::UShort => layout(vk::Format::R16_UINT, 2, 1),
        F::Short => layout(vk::Format::R16_UINT, 2, 1),
        F::UShortNormalized => layout(vk::Format::R16_UNORM, 2, 1),
        F::ShortNormalized => layout(vk::Format::R16_SNORM, 2, 1),
        F::Half => layout(vk::Format::R16_SFLOAT, 2, 1),

        F::UShort2 => layout(vk::Format::R16G16_UINT, 4, 2),
        F::Short2 => layout(vk::Format::R16G16_UINT, 4, 2),
        F::UShort2Normalized => layout(vk::Format::R16G16_UNORM, 4, 2),
        F::Short2Normalized => layout(vk::Format::R16G16_SNORM, 4, 2),
        F::Half2 => layout(vk::Format::R16G16_SFLOAT, 4, 2),

        F::UShort3 => layout(vk::Format::R16G16B16_UINT, 6, 3),
        F::Short3 => layout(vk::Format::R16G16B16_UINT, 6, 3),
        F::UShort3Normalized => layout(vk::Format::R16G16B16_UNORM, 6, 3),
        F::Short3Normalized => layout(vk::Format::R16G16B16_SNORM, 6, 3),
        F::Half3 => layout(vk::Format::R16G16B16_SFLOAT, 6, 3),

        F::UShort4 => layout(vk::Format::R16G16B16A16_UINT, 8, 4),
        F::Short4 => layout(vk::Format::R16G16B16A16_UINT, 8, 4),
        F::UShort4Normalized => layout(vk::Format::R16G16B16A16_UNORM, 8, 4),
        F::Short4Normalized => layout(vk::Format::R16G16B16A16_SNORM, 8, 4),
        F::Half4 => layout(vk::Format::R16G16B16A16_SFLOAT, 8, 4),

        F::Float => layout(vk::Format::R32_SFLOAT, 4, 1),
        F::Int => layout(vk::Format::R32_UINT, 4, 1),
        F::UInt => layout(vk::Format::R32_UINT, 4, 1),

        F::Float2 => layout(vk::Format::R32G32_SFLOAT, 8, 2),
        F::Int2 => layout(vk::Format::R32G32_UINT, 8, 2),
        F::UInt2 => layout(vk::Format::R32G32_UINT, 8, 2),

        F::Float3 => layout(vk::Format::R32G32B32_SFLOAT, 12, 3),
        F::Int3 => layout(vk::Format::R32G32B32_UINT, 12, 3),
        F::UInt3 => layout(vk::Format::R32G32B32_UINT, 12, 3),

        F::Float4 => layout(vk::Format::R32G32B32A32_SFLOAT, 16, 4),
        F::Int4 => layout(vk::Format::R32G32B32A32_UINT, 16, 4),
        F::UInt4 => layout(vk::Format::R32G32B32A32_UINT, 16, 4),

        F::Int1010102Normalized => layout(vk::Format::A2B10G10R10_SNORM_PACK32, 4, 4),
        F::UInt1010102Normalized => layout(vk::Format::A2B10G10R10_UNORM_PACK32, 4, 4),
        F::FloatRg11B10 => layout(vk::Format::B10G11R11_UFLOAT_PACK32, 4, 3),
        F::FloatRgb9E5 => layout(vk::Format::E5B9G9R9_UFLOAT_PACK32, 4, 3),
    }
}

/// The Vulkan format for an attribute, before any device-capability check.
pub fn vk_format(format: F) -> vk::Format {
    vertex_layout(format).vk
}

/// The attribute's footprint in the vertex buffer.
pub fn byte_size(format: F) -> u32 {
    vertex_layout(format).bytes
}

/// The signed Metal formats deliberately bound to unsigned Vulkan formats.
///
/// Enumerated rather than commented so the cross-repository ABI dependency has
/// a blast radius you can read. Every entry is correct only while metal2vulkan
/// emits unsigned integer stage inputs; see the module docs.
pub const SIGNED_AS_UNSIGNED: &[F] = &[
    F::Char,
    F::Char2,
    F::Char3,
    F::Char4,
    F::Short,
    F::Short2,
    F::Short3,
    F::Short4,
    F::Int,
    F::Int2,
    F::Int3,
    F::Int4,
];

/// `MTLVertexFormat` (SDK numeric values) → the engine's attribute format.
pub fn attribute_format(mtl: u32) -> Result<F, TranslateReason> {
    Ok(match mtl {
        1 => F::UChar2,
        2 => F::UChar3,
        3 => F::UChar4,
        4 => F::Char2,
        5 => F::Char3,
        6 => F::Char4,
        7 => F::UChar2Normalized,
        8 => F::UChar3Normalized,
        9 => F::UChar4Normalized,
        10 => F::Char2Normalized,
        11 => F::Char3Normalized,
        12 => F::Char4Normalized,
        13 => F::UShort2,
        14 => F::UShort3,
        15 => F::UShort4,
        16 => F::Short2,
        17 => F::Short3,
        18 => F::Short4,
        19 => F::UShort2Normalized,
        20 => F::UShort3Normalized,
        21 => F::UShort4Normalized,
        22 => F::Short2Normalized,
        23 => F::Short3Normalized,
        24 => F::Short4Normalized,
        25 => F::Half2,
        26 => F::Half3,
        27 => F::Half4,
        28 => F::Float,
        29 => F::Float2,
        30 => F::Float3,
        31 => F::Float4,
        32 => F::Int,
        33 => F::Int2,
        34 => F::Int3,
        35 => F::Int4,
        36 => F::UInt,
        37 => F::UInt2,
        38 => F::UInt3,
        39 => F::UInt4,
        40 => F::Int1010102Normalized,
        41 => F::UInt1010102Normalized,
        42 => F::UChar4NormalizedBgra,
        45 => F::UChar,
        46 => F::Char,
        47 => F::UCharNormalized,
        48 => F::CharNormalized,
        49 => F::UShort,
        50 => F::Short,
        51 => F::UShortNormalized,
        52 => F::ShortNormalized,
        53 => F::Half,
        54 => F::FloatRg11B10,
        55 => F::FloatRgb9E5,
        other => return Err(TranslateReason::UnknownVertexFormat(other)),
    })
}

/// A layout entry's declared `MTLVertexStepFunction` → engine step mode.
///
/// The serializer omits the field for Metal's default `PerVertex` behavior, so
/// absence is part of this translation rather than a caller-side fallback.
///
/// The SDK enum runs 0-4. Only 0-2 have a `VkVertexInputRate` — Vulkan has
/// `VERTEX` and `INSTANCE` and nothing else — so 3 (`PerPatch`) and 4
/// (`PerPatchControlPoint`) decline, but under their own reason rather than as
/// unrecognised values. They are recognised; this backend builds no
/// tessellation pipeline for them to belong to.
pub fn step_function(declared: Option<u32>) -> Result<VertexStepFunction, TranslateReason> {
    let Some(mtl) = declared else {
        return Ok(VertexStepFunction::PerVertex);
    };
    match mtl {
        0 => Ok(VertexStepFunction::Constant),
        1 => Ok(VertexStepFunction::PerVertex),
        2 => Ok(VertexStepFunction::PerInstance),
        3 | 4 => Err(TranslateReason::VertexStepFunctionPerPatch(mtl)),
        other => Err(TranslateReason::UnknownVertexStepFunction(other)),
    }
}

/// `VertexStepFunction` → the Vulkan input rate the binding is created with.
///
/// `Constant` has no Vulkan spelling of its own: Metal advances a constant-rate
/// attribute per instance with a divisor, so it lowers to `INSTANCE` and the
/// divisor carries the rest. The divisor is chosen beside the binding it
/// belongs to; this decides only the rate.
pub fn vk_input_rate(step: VertexStepFunction) -> vk::VertexInputRate {
    match step {
        VertexStepFunction::PerVertex => vk::VertexInputRate::VERTEX,
        VertexStepFunction::Constant | VertexStepFunction::PerInstance => {
            vk::VertexInputRate::INSTANCE
        }
    }
}

#[cfg(test)]
pub(super) const ALL_FORMATS: &[F] = &[
    F::UChar2,
    F::UChar3,
    F::UChar4,
    F::Char2,
    F::Char3,
    F::Char4,
    F::UChar2Normalized,
    F::UChar3Normalized,
    F::UChar4Normalized,
    F::Char2Normalized,
    F::Char3Normalized,
    F::Char4Normalized,
    F::UShort2,
    F::UShort3,
    F::UShort4,
    F::Short2,
    F::Short3,
    F::Short4,
    F::UShort2Normalized,
    F::UShort3Normalized,
    F::UShort4Normalized,
    F::Short2Normalized,
    F::Short3Normalized,
    F::Short4Normalized,
    F::Half2,
    F::Half3,
    F::Half4,
    F::Float,
    F::Float2,
    F::Float3,
    F::Float4,
    F::Int,
    F::Int2,
    F::Int3,
    F::Int4,
    F::UInt,
    F::UInt2,
    F::UInt3,
    F::UInt4,
    F::Int1010102Normalized,
    F::UInt1010102Normalized,
    F::UChar4NormalizedBgra,
    F::UChar,
    F::Char,
    F::UCharNormalized,
    F::CharNormalized,
    F::UShort,
    F::Short,
    F::UShortNormalized,
    F::ShortNormalized,
    F::Half,
    F::FloatRg11B10,
    F::FloatRgb9E5,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The SDK enum is contiguous except for the gap at 43/44 that Apple left
    /// when it moved the single-component formats to 45+. Every defined value
    /// maps; the gap and everything past the end decline by name.
    #[test]
    fn every_sdk_vertex_format_value_maps() {
        let mut seen = Vec::new();
        for mtl in 0..=64u32 {
            match attribute_format(mtl) {
                Ok(f) => seen.push((mtl, f)),
                Err(e) => assert_eq!(e, TranslateReason::UnknownVertexFormat(mtl)),
            }
        }
        assert_eq!(seen.len(), ALL_FORMATS.len());
        // 0 is MTLVertexFormatInvalid; 43/44 are the reserved gap.
        for gap in [0u32, 43, 44, 56, 1000] {
            assert_eq!(
                attribute_format(gap).unwrap_err(),
                TranslateReason::UnknownVertexFormat(gap),
                "{gap}"
            );
        }
        // No two Metal values collapse onto the same engine format by accident
        // *except* the signed/unsigned pairs, which do so on purpose.
        let mut formats: Vec<F> = seen.iter().map(|(_, f)| *f).collect();
        formats.sort_by_key(|f| format!("{f:?}"));
        formats.dedup();
        assert_eq!(formats.len(), ALL_FORMATS.len());
    }

    #[test]
    fn every_vertex_step_function_maps_and_absence_is_per_vertex() {
        assert_eq!(step_function(None).unwrap(), VertexStepFunction::PerVertex);
        assert_eq!(
            step_function(Some(0)).unwrap(),
            VertexStepFunction::Constant
        );
        assert_eq!(
            step_function(Some(1)).unwrap(),
            VertexStepFunction::PerVertex
        );
        assert_eq!(
            step_function(Some(2)).unwrap(),
            VertexStepFunction::PerInstance
        );
        assert_eq!(
            step_function(Some(5)).unwrap_err(),
            TranslateReason::UnknownVertexStepFunction(5)
        );
    }

    /// Every ordinal this backend accepts survives the round trip back to the
    /// wire value it came from.
    ///
    /// [`VertexStepFunction::mtl_ordinal`] exists so a rule stated over the
    /// guest's ordinal can be asked on this side — the step/rate pair in
    /// `contract::vertex_step` is the one that does — and a rule asked through
    /// an inverse that is not an inverse is a rule asked about a different
    /// attribute. The three accepted ordinals are named from the contract here
    /// rather than spelled again, so this also pins that the `match` above
    /// agrees with the declaration.
    #[test]
    fn an_accepted_step_function_round_trips_to_its_own_ordinal() {
        use crate::contract::vertex_step as step;
        for ordinal in [
            step::MTL_VERTEX_STEP_FUNCTION_CONSTANT,
            step::MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            step::MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
        ] {
            let translated = step_function(Some(ordinal)).expect("an accepted ordinal");
            assert_eq!(
                translated.mtl_ordinal(),
                ordinal,
                "{translated:?} came from {ordinal}"
            );
        }
    }

    /// The two tessellation step rates decline under their own reason, and the
    /// first value genuinely off the end of the SDK enum declines under the
    /// other.
    ///
    /// This test asserted `UnknownVertexStepFunction(3)` before, which encoded
    /// the wrong end of `MTLVertexStepFunction` as the intended behaviour: the
    /// enum runs to `PerPatchControlPoint = 4`, so 3 and 4 are declared values
    /// this backend recognises and cannot spell, not values it fails to
    /// recognise. Reading a boot's log, the two want different answers — one is
    /// "the guest ran a tessellation pipeline", the other is "something is
    /// wrong upstream of here".
    #[test]
    fn the_two_tessellation_step_rates_decline_by_their_own_name() {
        for mtl in [3u32, 4] {
            assert_eq!(
                step_function(Some(mtl)).unwrap_err(),
                TranslateReason::VertexStepFunctionPerPatch(mtl),
                "MTLVertexStepFunction {mtl}"
            );
        }
        use crate::observe::Decline as _;
        assert_eq!(
            TranslateReason::VertexStepFunctionPerPatch(3).slug(),
            "vertex_step_function_per_patch"
        );
        assert_ne!(
            TranslateReason::VertexStepFunctionPerPatch(3).slug(),
            TranslateReason::UnknownVertexStepFunction(5).slug()
        );
        // Absence is its own state now rather than a flag beside a word, so a
        // record that never carried the field cannot reach either refusal.
        assert_eq!(step_function(None).unwrap(), VertexStepFunction::PerVertex);
    }

    /// **L2's co-location invariant.** The byte size must equal the Vulkan
    /// format's own texel size for every arm. This is what the two separate
    /// matches could not guarantee: here a mismatched arm cannot compile past
    /// this table without failing.
    #[test]
    fn byte_size_matches_the_vulkan_format_it_sits_beside() {
        for f in ALL_FORMATS {
            let l = vertex_layout(*f);
            assert_eq!(
                l.bytes,
                vk_format_texel_size(l.vk),
                "{f:?}: {} bytes beside {:?}",
                l.bytes,
                l.vk
            );
            assert!(l.components >= 1 && l.components <= 4, "{f:?}");
        }
    }

    /// Independent size table, written from the Vulkan format names rather than
    /// from the code under test — otherwise the invariant above just agrees
    /// with itself.
    fn vk_format_texel_size(f: vk::Format) -> u32 {
        match f {
            vk::Format::R8_UINT | vk::Format::R8_UNORM | vk::Format::R8_SNORM => 1,
            vk::Format::R8G8_UINT | vk::Format::R8G8_UNORM | vk::Format::R8G8_SNORM => 2,
            vk::Format::R8G8B8_UINT | vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SNORM => 3,
            vk::Format::R8G8B8A8_UINT
            | vk::Format::R8G8B8A8_UNORM
            | vk::Format::R8G8B8A8_SNORM
            | vk::Format::B8G8R8A8_UNORM => 4,
            vk::Format::R16_UINT
            | vk::Format::R16_UNORM
            | vk::Format::R16_SNORM
            | vk::Format::R16_SFLOAT => 2,
            vk::Format::R16G16_UINT
            | vk::Format::R16G16_UNORM
            | vk::Format::R16G16_SNORM
            | vk::Format::R16G16_SFLOAT => 4,
            vk::Format::R16G16B16_UINT
            | vk::Format::R16G16B16_UNORM
            | vk::Format::R16G16B16_SNORM
            | vk::Format::R16G16B16_SFLOAT => 6,
            vk::Format::R16G16B16A16_UINT
            | vk::Format::R16G16B16A16_UNORM
            | vk::Format::R16G16B16A16_SNORM
            | vk::Format::R16G16B16A16_SFLOAT => 8,
            vk::Format::R32_UINT | vk::Format::R32_SFLOAT => 4,
            vk::Format::R32G32_UINT | vk::Format::R32G32_SFLOAT => 8,
            vk::Format::R32G32B32_UINT | vk::Format::R32G32B32_SFLOAT => 12,
            vk::Format::R32G32B32A32_UINT | vk::Format::R32G32B32A32_SFLOAT => 16,
            vk::Format::A2B10G10R10_SNORM_PACK32
            | vk::Format::A2B10G10R10_UNORM_PACK32
            | vk::Format::B10G11R11_UFLOAT_PACK32
            | vk::Format::E5B9G9R9_UFLOAT_PACK32 => 4,
            other => panic!("no size for {other:?} — add it beside the layout arm"),
        }
    }

    /// The signed arms bind unsigned formats, deliberately, and the list of
    /// them is exactly the arms that do so.
    #[test]
    fn the_signed_arms_are_the_ones_bound_to_unsigned_formats() {
        for f in SIGNED_AS_UNSIGNED {
            let vk = vk_format(*f);
            let name = format!("{vk:?}");
            assert!(
                name.contains("UINT"),
                "{f:?} is in the signed-as-unsigned list but maps to {vk:?}"
            );
        }
        // Normalized signed formats are NOT in the list — they map to real
        // SNORM formats, whose numeric type is float, not int, so the emitter's
        // integer signedness does not apply to them.
        for f in [F::CharNormalized, F::Char2Normalized, F::Short4Normalized] {
            assert!(!SIGNED_AS_UNSIGNED.contains(&f));
            assert!(format!("{:?}", vk_format(f)).contains("SNORM"));
        }
        assert_eq!(SIGNED_AS_UNSIGNED.len(), 12);
    }

    /// The 32-bit signed arms are bit-exact: same width, no extension, so the
    /// shader body's signed instructions read the intended value.
    #[test]
    fn the_32_bit_signed_arms_are_bit_exact() {
        for (signed, unsigned) in [
            (F::Int, F::UInt),
            (F::Int2, F::UInt2),
            (F::Int3, F::UInt3),
            (F::Int4, F::UInt4),
        ] {
            assert_eq!(vertex_layout(signed), vertex_layout(unsigned));
        }
    }
}
