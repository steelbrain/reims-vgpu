//! `MTLBlendFactor` / `MTLBlendOperation` → the engine's blend state →
//! `VkBlendFactor` / `VkBlendOp`.
//!
//! Both halves of the crossing live here: the wire value → engine enum decode,
//! and the engine enum → Vulkan spelling. Keeping them apart is how a blend
//! factor ends up decoded on one path and spelled differently on another.

use ash::vk;

use super::reason::TranslateReason;
use crate::engine::{BlendFactor, BlendOp, BlendStateResource};
use reims_vgpu_protocol::resource::{
    ColorWriteMask, PipelineColorAttachment, MTL_COLOR_WRITE_MASK_ALPHA, MTL_COLOR_WRITE_MASK_BLUE,
    MTL_COLOR_WRITE_MASK_GREEN, MTL_COLOR_WRITE_MASK_RED,
};

/// `MTLBlendFactor` (SDK numeric values, Metal header order).
///
/// The table used to stop at 14 and its test asserted 15 as "past the end".
/// It is not: `MTLRenderPipeline.h` on the macOS 26 SDK runs to
/// `MTLBlendFactorOneMinusSource1Alpha = 18`, so 14 is the fifteenth of
/// nineteen, so the four dual-source factors were refused device-wide.
///
/// 15-18 are the dual-source factors, which read the fragment shader's second
/// colour output. Vulkan spells them `SRC1_*` and gates them behind
/// `VkPhysicalDeviceFeatures::dualSrcBlend`; naming one in a pipeline without
/// that feature is invalid, so the *capability* question is asked where the
/// pipeline is built (`caches::get_or_create_pipeline`) rather than here.
/// Translation is unconditional: whether the guest asked for a dual-source
/// blend and whether this host can run one are two different facts and
/// collapsing them would report a capability gap as a decode failure.
///
/// 19 (`MTLBlendFactorUnspecialized`) is deliberately absent. It is Metal 4's
/// "resolve this at specialization time" sentinel rather than a blend factor,
/// and nothing in this device performs that resolution.
pub fn factor(mtl: u32) -> Result<BlendFactor, TranslateReason> {
    reims_vgpu_protocol::blend_factor(mtl).map_err(|_| TranslateReason::UnknownBlendFactor(mtl))
}

/// `MTLBlendOperation` (SDK numeric values, Metal header order).
///
/// **The enum runs 0-5, not 0-4**, and the fifth is refused on purpose:
/// `MTLRenderPipeline.h` on the macOS 26 SDK declares
/// `MTLBlendOperationUnspecialized = 5`. Like [`factor`]'s
/// `MTLBlendFactorUnspecialized = 19` and `MTLColorWriteMaskUnspecialized =
/// 0x10` beside them, it is Metal 4's "resolve this at specialization time"
/// sentinel rather than an operation, and nothing in this device performs that
/// resolution — so there is nothing for a guest to lose by its refusal.
///
/// Saying so here is the point. Refusing 5 without a word about it leaves the
/// converter and its test in the exact shape that hid the dual-source blend
/// gap: a range mapped from 0 to N and `N + 1` asserted to decline, with
/// nothing distinguishing "one past the end of Apple's enum" from "a declared
/// Apple value this device turns down". The first is a bound; the second is a
/// decision, and only a decision can be wrong.
pub fn operation(mtl: u32) -> Result<BlendOp, TranslateReason> {
    reims_vgpu_protocol::blend_operation(mtl)
        .map_err(|_| TranslateReason::UnknownBlendOperation(mtl))
}

/// A whole decoded render-pipeline colour-attachment blend descriptor.
///
/// Fails on the first unrepresentable component rather than substituting a
/// default for it — a blend that silently becomes `ONE, ZERO` is a rendering
/// bug with no log line.
///
/// Takes the decoded attachment rather than its six factor/op ordinals: both
/// production callers hold one, the six are all `u32` and adjacent, and the RGB
/// and alpha halves are three-for-three interchangeable — a swap between them
/// produces a valid blend state that blends the wrong channel set, which no
/// decline can report because nothing was out of contract.
pub fn state(a: &PipelineColorAttachment) -> Result<BlendStateResource, TranslateReason> {
    reims_vgpu_protocol::blend_state(a).map_err(|reason| match reason {
        reims_vgpu_protocol::PipelineStateDecodeError::BlendFactor(value) => {
            TranslateReason::UnknownBlendFactor(value)
        }
        reims_vgpu_protocol::PipelineStateDecodeError::BlendOperation(value) => {
            TranslateReason::UnknownBlendOperation(value)
        }
        _ => unreachable!("blend_state returns only blend decode errors"),
    })
}

/// `MTLColorWriteMask` → `VkColorComponentFlags`.
///
/// Metal's bits run alpha-first from the low end (`alpha = 1 << 0` …
/// `red = 1 << 3`); Vulkan's run red-first (`R = 1 << 0` … `A = 1 << 3`). The
/// two are bit-reversed over four bits, not equal, so a straight cast would
/// swap red and alpha and leave green and blue exchanged — a mask asking for
/// alpha-only would write red-only.
///
/// Total over the mask's range by construction: the input is `ColorWriteMask`,
/// whose only producer is the decoder, and the decoder refuses anything above
/// `MTLColorWriteMaskAll` by name. Bits above the fourth are ignored here
/// rather than declined a second time.
pub fn vk_color_write_mask(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let bits = mask.bits();
    let mut out = vk::ColorComponentFlags::empty();
    if bits & MTL_COLOR_WRITE_MASK_RED != 0 {
        out |= vk::ColorComponentFlags::R;
    }
    if bits & MTL_COLOR_WRITE_MASK_GREEN != 0 {
        out |= vk::ColorComponentFlags::G;
    }
    if bits & MTL_COLOR_WRITE_MASK_BLUE != 0 {
        out |= vk::ColorComponentFlags::B;
    }
    if bits & MTL_COLOR_WRITE_MASK_ALPHA != 0 {
        out |= vk::ColorComponentFlags::A;
    }
    out
}

pub fn vk_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        BlendFactor::ConstantColor => vk::BlendFactor::CONSTANT_COLOR,
        BlendFactor::OneMinusConstantColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        BlendFactor::ConstantAlpha => vk::BlendFactor::CONSTANT_ALPHA,
        BlendFactor::OneMinusConstantAlpha => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
        BlendFactor::Src1Color => vk::BlendFactor::SRC1_COLOR,
        BlendFactor::OneMinusSrc1Color => vk::BlendFactor::ONE_MINUS_SRC1_COLOR,
        BlendFactor::Src1Alpha => vk::BlendFactor::SRC1_ALPHA,
        BlendFactor::OneMinusSrc1Alpha => vk::BlendFactor::ONE_MINUS_SRC1_ALPHA,
    }
}

pub fn vk_operation(op: BlendOp) -> vk::BlendOp {
    match op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's blend enums are dense from 0, so the whole range maps — and the
    /// two values above each range decline for **two different reasons**.
    ///
    /// The factor run ends at **18**, not 14. This test asserted `factor(15)`
    /// was past it, which is what let the four dual-source factors read as a
    /// covered range for as long as they did: 15 is
    /// `MTLBlendFactorSource1Color`, and the run ends at
    /// `OneMinusSource1Alpha = 18`.
    ///
    /// Fixing the bound alone would leave the same trap one notch along, so the
    /// two refusals are now asserted apart. `factor(19)` and `operation(5)` are
    /// **declared SDK values** — the `Unspecialized` specialization sentinels —
    /// refused because this device resolves nothing at specialization time.
    /// `factor(20)` and `operation(6)` are genuinely past the end of Apple's
    /// enums. A test that only checks the first value that declines cannot tell
    /// those apart, and cannot notice when Apple adds a real member where the
    /// sentinel used to be the boundary: that is precisely what happened between
    /// `MTLBlendFactorOneMinusBlendAlpha = 14` and the dual-source four.
    #[test]
    fn the_blend_enums_are_total_over_their_sdk_range() {
        for mtl in 0..=18u32 {
            assert!(factor(mtl).is_ok(), "MTLBlendFactor {mtl}");
        }
        for mtl in 0..=4u32 {
            assert!(operation(mtl).is_ok(), "MTLBlendOperation {mtl}");
        }

        // Declared by Apple, refused by decision.
        assert_eq!(
            factor(19).unwrap_err(),
            TranslateReason::UnknownBlendFactor(19),
            "MTLBlendFactorUnspecialized is declared; refusing it is a decision \
             this module documents, not a range bound"
        );
        assert_eq!(
            operation(5).unwrap_err(),
            TranslateReason::UnknownBlendOperation(5),
            "MTLBlendOperationUnspecialized is declared; refusing it is a \
             decision this module documents, not a range bound"
        );

        // Past the end of Apple's enums.
        assert_eq!(
            factor(20).unwrap_err(),
            TranslateReason::UnknownBlendFactor(20)
        );
        assert_eq!(
            operation(6).unwrap_err(),
            TranslateReason::UnknownBlendOperation(6)
        );
    }

    /// The four dual-source factors map to Vulkan's `SRC1_*` and report
    /// themselves as needing `dualSrcBlend`; the fifteen below them do not.
    ///
    /// The split has to be exact in both directions. Marking a plain factor
    /// dual-source would decline a pipeline every compositor builds on any host
    /// without the feature; missing one would build a pipeline Vulkan rejects,
    /// which is the ungated-bind shape `caps::device_features` exists to stop.
    #[test]
    fn only_the_four_dual_source_factors_ask_for_the_feature() {
        for mtl in 0..=14u32 {
            assert!(
                !factor(mtl).unwrap().is_dual_source(),
                "MTLBlendFactor {mtl} is not dual-source"
            );
        }
        for mtl in 15..=18u32 {
            assert!(
                factor(mtl).unwrap().is_dual_source(),
                "MTLBlendFactor {mtl} is dual-source"
            );
        }
        assert_eq!(vk_factor(factor(15).unwrap()), vk::BlendFactor::SRC1_COLOR);
        assert_eq!(
            vk_factor(factor(16).unwrap()),
            vk::BlendFactor::ONE_MINUS_SRC1_COLOR
        );
        assert_eq!(vk_factor(factor(17).unwrap()), vk::BlendFactor::SRC1_ALPHA);
        assert_eq!(
            vk_factor(factor(18).unwrap()),
            vk::BlendFactor::ONE_MINUS_SRC1_ALPHA
        );
    }

    /// Every engine factor reaches a distinct Vulkan factor — two collapsing
    /// onto one would silently change how a surface composites.
    #[test]
    fn every_blend_factor_has_a_distinct_vulkan_spelling() {
        let all: Vec<BlendFactor> = (0..=18).map(|m| factor(m).unwrap()).collect();
        let mut vks: Vec<i32> = all.iter().map(|f| vk_factor(*f).as_raw()).collect();
        vks.sort_unstable();
        let before = vks.len();
        vks.dedup();
        assert_eq!(before, vks.len());

        let mut ops: Vec<i32> = (0..=4)
            .map(|m| vk_operation(operation(m).unwrap()).as_raw())
            .collect();
        ops.sort_unstable();
        let before = ops.len();
        ops.dedup();
        assert_eq!(before, ops.len());
    }

    /// The two enums share an ordering with Metal's headers for the first
    /// several values; spot-check the ones a transcription slip would swap.
    #[test]
    fn the_load_bearing_arms_match_the_metal_header() {
        assert_eq!(vk_factor(factor(4).unwrap()), vk::BlendFactor::SRC_ALPHA);
        assert_eq!(
            vk_factor(factor(5).unwrap()),
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        );
        assert_eq!(
            vk_factor(factor(10).unwrap()),
            vk::BlendFactor::SRC_ALPHA_SATURATE
        );
        assert_eq!(
            vk_operation(operation(2).unwrap()),
            vk::BlendOp::REVERSE_SUBTRACT
        );
    }

    /// Metal's mask bits are alpha-first and Vulkan's are red-first, so the
    /// two are bit-reversed over four bits. A cast would swap red with alpha
    /// and green with blue, and the mask that motivated decoding this field at
    /// all — alpha-only — would come out as red-only, which writes colour and
    /// drops the coverage the guest was punching in.
    #[test]
    fn the_metal_and_vulkan_write_mask_bit_orders_are_reversed_not_equal() {
        use reims_vgpu_protocol::resource::{
            MTL_COLOR_WRITE_MASK_ALL, MTL_COLOR_WRITE_MASK_BLUE, MTL_COLOR_WRITE_MASK_GREEN,
            MTL_COLOR_WRITE_MASK_NONE,
        };
        let of = |bits: u32| vk_color_write_mask(ColorWriteMask::new(bits).unwrap());

        assert_eq!(of(MTL_COLOR_WRITE_MASK_ALPHA), vk::ColorComponentFlags::A);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_RED), vk::ColorComponentFlags::R);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_GREEN), vk::ColorComponentFlags::G);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_BLUE), vk::ColorComponentFlags::B);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_ALL), vk::ColorComponentFlags::RGBA);
        assert_eq!(
            of(MTL_COLOR_WRITE_MASK_NONE),
            vk::ColorComponentFlags::empty()
        );
        // The default is `all`, which is what an entry with no tag means.
        assert_eq!(
            vk_color_write_mask(ColorWriteMask::default()),
            vk::ColorComponentFlags::RGBA
        );
        // A straight cast would agree on `all` and `none` and disagree on
        // every single-channel mask; assert the disagreement so a later
        // "simplification" to `from_raw(bits)` fails here.
        assert_ne!(
            of(MTL_COLOR_WRITE_MASK_ALPHA),
            vk::ColorComponentFlags::from_raw(MTL_COLOR_WRITE_MASK_ALPHA)
        );
        // Every mask in range maps injectively — two collapsing onto one would
        // silently merge distinct pipelines.
        let mut seen: Vec<u32> = (0..=MTL_COLOR_WRITE_MASK_ALL)
            .map(|m| of(m).as_raw())
            .collect();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(before, seen.len());
    }

    /// A whole descriptor fails on its first bad component instead of
    /// substituting a default — the difference between a visible decline and a
    /// surface that composites wrong for the rest of the boot.
    #[test]
    fn a_bad_component_fails_the_whole_descriptor() {
        // One, OneMinusSrcAlpha, Add on both halves.
        let attach =
            |src_rgb, dst_rgb, op_rgb, src_alpha, dst_alpha, op_alpha| PipelineColorAttachment {
                src_rgb,
                dst_rgb,
                op_rgb,
                src_alpha,
                dst_alpha,
                op_alpha,
                ..Default::default()
            };
        let ok = state(&attach(1, 5, 0, 1, 5, 0)).unwrap();
        assert_eq!(ok.src_color, BlendFactor::One);
        assert_eq!(ok.dst_color, BlendFactor::OneMinusSrcAlpha);
        assert_eq!(ok.color_op, BlendOp::Add);
        assert_eq!(
            state(&attach(1, 99, 0, 1, 5, 0)).unwrap_err(),
            TranslateReason::UnknownBlendFactor(99)
        );
        assert_eq!(
            state(&attach(1, 5, 0, 1, 5, 77)).unwrap_err(),
            TranslateReason::UnknownBlendOperation(77)
        );
    }

    /// The RGB and alpha halves land in the fields of their own names.
    ///
    /// The six ordinals are interchangeable `u32`s three-for-three, so a swap
    /// between the halves yields a perfectly valid blend state that blends the
    /// wrong channel set — no decline, no log line. Distinct factors on each
    /// half is the only arrangement that can see it.
    #[test]
    fn the_rgb_and_alpha_halves_do_not_cross() {
        let b = state(&PipelineColorAttachment {
            src_rgb: 1,   // One
            dst_rgb: 5,   // OneMinusSrcAlpha
            op_rgb: 0,    // Add
            src_alpha: 4, // SrcAlpha
            dst_alpha: 0, // Zero
            op_alpha: 1,  // Subtract
            ..Default::default()
        })
        .unwrap();
        assert_eq!(b.src_color, BlendFactor::One);
        assert_eq!(b.dst_color, BlendFactor::OneMinusSrcAlpha);
        assert_eq!(b.color_op, BlendOp::Add);
        assert_eq!(b.src_alpha, BlendFactor::SrcAlpha);
        assert_eq!(b.dst_alpha, BlendFactor::Zero);
        assert_eq!(b.alpha_op, BlendOp::Subtract);
    }
}
