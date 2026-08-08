//! Guest ordinals to Metal enums, checked.
//!
//! Every value that reaches this module was decoded out of the guest's command
//! stream, so it is an arbitrary `u32`. The `MTL*` types are fieldless
//! `#[repr(u64)]` enums, and producing one whose discriminant is not a declared
//! variant is **undefined behaviour**, not a decode error — the same rule
//! `reims-vgpu-wire`'s invariant 4 states for wire structs, and the reason that
//! crate stores raw scalars and exposes fallible accessors. This module is the
//! device-side half of it: name every variant, return `None` for anything else,
//! and let the caller turn that into a typed refusal.
//!
//! It replaced 28 `transmute::<u64, MTL*>` calls. Two things that audit found
//! are worth keeping, because both defeat the obvious repair:
//!
//! - **A `<= max` range check is not sufficient, because two of these enums have
//!   holes.** `MTLVertexFormat` and `MTLAttributeFormat` run 0–42 and then
//!   resume at 45: Apple left 43 and 44 unassigned between
//!   `UChar4Normalized_BGRA` and `UChar`. A bound of "not above the last
//!   variant" admits both, and `stage_input`'s attribute-format guard did.
//! - **Five of the sites had no check at all.** The two vertex-descriptor
//!   builders (`backend::metal::render::make_vertex_descriptor` and
//!   `runtime::icb::metal_vertex_descriptor_from_attrs_for_draw`) transmuted a
//!   type-7 pipeline descriptor's `format` and `step_function` words straight
//!   through, and those are guest bytes with no producer-side clamp anywhere.
//!
//! ## Where the variant lists come from
//!
//! The `metal` crate's own enum declarations, which are what define the Rust
//! type — a value it does not declare has no legal representation here however
//! well Apple documents it. Those declarations were checked against the Metal
//! SDK headers on this host (macOS 26 SDK, `MTLVertexDescriptor.h`,
//! `MTLStageInputOutputDescriptor.h`, `MTLRenderPipeline.h`,
//! `MTLRenderCommandEncoder.h`, `MTLRenderPass.h`, `MTLDepthStencil.h`,
//! `MTLArgument.h`), and they agree everywhere except at the top of four:
//!
//! | enum | SDK tail | `metal` 0.33 tail |
//! |---|---|---|
//! | `MTLVertexFormat` | `FloatRG11B10` 54, `FloatRGB9E5` 55 | stops at `Half` 53 |
//! | `MTLAttributeFormat` | `FloatRG11B10` 54, `FloatRGB9E5` 55 | stops at `Half` 53 |
//! | `MTLBlendFactor` | `Unspecialized` 19 | stops at `OneMinusSource1Alpha` 18 |
//! | `MTLBlendOperation` | `Unspecialized` 5 | stops at `Max` 4 |
//!
//! So a guest asking for one of those six values is declined here rather than
//! executed. That is a narrowing against what Metal itself would accept, and it
//! is deliberate: the alternative is undefined behaviour, and there is no third
//! option inside `metal`'s type. The refusal is counted at each call site, so a
//! non-zero reading is the measured argument for reaching those values the way
//! [`crate::backend::metal::raw_metal`] reaches other unwrapped selectors — by
//! sending the setter a raw `u64` and never materializing an enum at all. Until
//! one fires, that is four wrappers for traffic nobody has seen.

use metal::{
    MTLAttributeFormat, MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode,
    MTLDepthClipMode, MTLIndexType, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLSamplerAddressMode, MTLSamplerBorderColor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLStencilOperation, MTLStepFunction, MTLStoreAction, MTLTriangleFillMode, MTLVertexFormat,
    MTLVertexStepFunction, MTLVisibilityResultMode, MTLWinding,
};

/// Define a checked ordinal to enum conversion, and prove it at compile time.
///
/// The generated `const` block asserts both directions against the variant list
/// given here: every listed variant converts back to itself, and no ordinal that
/// is *not* listed converts at all, sweeping to four past the highest listed
/// discriminant so interior holes and the upper edge are both covered.
///
/// # The trailing `apple_numbers_them_from_zero;` clause
///
/// Everything above is stated in terms of `<$ty>::$variant as u32` and nothing
/// else, so **it names no number.** It proves a table agrees with itself, which
/// leaves the accepted *set* — the ordinals a guest may send — defined entirely
/// by whatever discriminants the `metal` crate happens to assign. A crate bump
/// that renumbers one of these enums keeps every assertion green while moving
/// the set: the Metal arm would start refusing an ordinal Apple declares (and
/// the Vulkan arm still accepts, since `translate::raster` spells Apple's
/// numbers itself), and nothing in the tree would say so.
///
/// The clause closes that for the tables whose variants Apple numbers `0..n`
/// with no holes, in the order they are listed here. It asserts each variant's
/// discriminant equals its index, which is the whole accepted set stated as one
/// claim. Add it only after reading the SDK header — it is an assertion about
/// Apple's numbering that the compiler checks against `metal`'s, so writing it
/// on a table with a hole (`vertex_format`, `attribute_format`) is false, and
/// writing it on a table listed out of Apple's order pins the wrong thing.
///
/// # What this cannot catch
///
/// **A renaming.** The clause constrains discriminants, not names, and this
/// crate version is measured to put six of `MTLStepFunction`'s nine names on
/// the wrong numbers while numbering the enum `0..8` contiguously — so a table
/// of it would satisfy the clause and still be wrong. That enum is converted
/// through `STEP_FUNCTION_BY_ORDINAL` below for exactly this reason.
///
/// **A variant the `metal` crate declares and this invocation omits.** The list
/// is the only statement of the accepted set, so an omitted variant is absent
/// from the conversion and from the sweep alike: the ordinal reads as undeclared,
/// `$fn_name` returns `None` for it, and the assertion that undeclared ordinals
/// return `None` passes. The guest's value silently becomes a refusal.
///
/// This was verified by deleting `Half3` from `vertex_format`'s list and
/// watching the build stay green. It is not fixable here — Rust cannot enumerate
/// a foreign enum's variants — so the guard is the module doc above: the lists
/// were read off `metal` 0.33's own declarations, and a crate bump means
/// re-reading them. An earlier version of this doc claimed omission "fails its
/// own test"; it never did.
macro_rules! checked_ordinal {
    (
        $(#[$outer:meta])*
        fn $fn_name:ident -> $ty:ty;
        [ $($variant:ident),+ $(,)? ]
        apple_numbers_them_from_zero;
    ) => {
        checked_ordinal! {
            $(#[$outer])*
            fn $fn_name -> $ty;
            [ $($variant),+ ]
        }

        // Each variant sits at the number its position in the list claims, which
        // for a hole-free enum listed in Apple's order is the accepted set
        // stated once. Without it the set is whatever `metal` says it is, and a
        // renumbering there is a silent cross-arm divergence rather than a build
        // failure — see the macro's own doc.
        const _: () = {
            let declared = [ $(<$ty>::$variant as u32),+ ];
            let mut index = 0usize;
            while index < declared.len() {
                assert!(
                    declared[index] == index as u32,
                    concat!(
                        stringify!($fn_name),
                        " has a variant off the number Apple assigned it",
                    ),
                );
                index += 1;
            }
        };
    };
    (
        $(#[$outer:meta])*
        fn $fn_name:ident -> $ty:ty;
        [ $($variant:ident),+ $(,)? ]
    ) => {
        $(#[$outer])*
        pub(crate) const fn $fn_name(ordinal: u32) -> Option<$ty> {
            $(
                if ordinal == <$ty>::$variant as u32 {
                    return Some(<$ty>::$variant);
                }
            )+
            None
        }

        // The same sweep the generated `#[test]` used to run, as a `const`
        // block, and `$fn_name` is a `const fn` so that it can be. The reason is
        // the one `super::constants` spells out: this module is
        // `backend-metal`-gated, so a `#[cfg(test)]` test in it is compiled out
        // of the Vulkan arm and its `--lib` suite runs on Apple hosts only.
        // Seventeen tables' worth of "no undeclared ordinal converts" was
        // therefore checked on no machine anybody edits this code from — and
        // what it stands between is a guest `u32` and a `transmute` into a
        // `#[repr(u64)]` enum, which is undefined behaviour rather than a decode
        // error. `rustc` evaluates this on every arm that compiles the file,
        // including the cross-compiled Metal clippy run `AGENTS.md` requires
        // from Linux.
        //
        // Ordinals are compared rather than values: `metal` derives `PartialEq`
        // on most of these enums and not all (`MTLStoreAction` and
        // `MTLLoadAction` have no derive at all), and the ordinal is what the
        // assertion is about.
        const _: () = {
            // Every declared variant converts back to itself.
            $(
                assert!(
                    match $fn_name(<$ty>::$variant as u32) {
                        Some(got) => got as u32 == <$ty>::$variant as u32,
                        None => false,
                    },
                    concat!(stringify!($fn_name), " rejected one of its own variants"),
                );
            )+

            // Everything up to four past the last declared variant, so the
            // run's interior holes and its upper edge are both covered.
            let mut ceiling = 0u32;
            $(
                if (<$ty>::$variant as u32) > ceiling {
                    ceiling = <$ty>::$variant as u32;
                }
            )+
            ceiling = ceiling.saturating_add(4);

            let mut ordinal = 0u32;
            while ordinal <= ceiling {
                let mut declared = false;
                $(
                    if ordinal == <$ty>::$variant as u32 {
                        declared = true;
                    }
                )+
                assert!(
                    declared || $fn_name(ordinal).is_none(),
                    concat!(stringify!($fn_name), " accepted an undeclared ordinal"),
                );
                ordinal += 1;
            }

            assert!($fn_name(u32::MAX).is_none());
        };
    };
}

checked_ordinal! {
    /// `MTLVertexFormat` for a vertex-descriptor attribute.
    ///
    /// Declines 43 and 44 — the gap Apple left between `UChar4Normalized_BGRA`
    /// and `UChar` — and 54/55, which the SDK declares and `metal` does not.
    fn vertex_format -> MTLVertexFormat;
    [
        Invalid, UChar2, UChar3, UChar4, Char2, Char3, Char4,
        UChar2Normalized, UChar3Normalized, UChar4Normalized,
        Char2Normalized, Char3Normalized, Char4Normalized,
        UShort2, UShort3, UShort4, Short2, Short3, Short4,
        UShort2Normalized, UShort3Normalized, UShort4Normalized,
        Short2Normalized, Short3Normalized, Short4Normalized,
        Half2, Half3, Half4, Float, Float2, Float3, Float4,
        Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4,
        Int1010102Normalized, UInt1010102Normalized, UChar4Normalized_BGRA,
        UChar, Char, UCharNormalized, CharNormalized,
        UShort, Short, UShortNormalized, ShortNormalized, Half,
    ]
}

checked_ordinal! {
    /// `MTLAttributeFormat` for a compute stage-input attribute.
    ///
    /// The same numbering as [`vertex_format`], including the 43/44 hole; Metal
    /// declares the two enums separately and this device must not assume they
    /// stay in step.
    fn attribute_format -> MTLAttributeFormat;
    [
        Invalid, UChar2, UChar3, UChar4, Char2, Char3, Char4,
        UChar2Normalized, UChar3Normalized, UChar4Normalized,
        Char2Normalized, Char3Normalized, Char4Normalized,
        UShort2, UShort3, UShort4, Short2, Short3, Short4,
        UShort2Normalized, UShort3Normalized, UShort4Normalized,
        Short2Normalized, Short3Normalized, Short4Normalized,
        Half2, Half3, Half4, Float, Float2, Float3, Float4,
        Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4,
        Int1010102Normalized, UInt1010102Normalized, UChar4Normalized_BGRA,
        UChar, Char, UCharNormalized, CharNormalized,
        UShort, Short, UShortNormalized, ShortNormalized, Half,
    ]
}

checked_ordinal! {
    /// `MTLVertexStepFunction` for a vertex-descriptor buffer layout.
    fn vertex_step_function -> MTLVertexStepFunction;
    [Constant, PerVertex, PerInstance, PerPatch, PerPatchControlPoint]
    apple_numbers_them_from_zero;
}

// The clause above pins the five discriminants to `0..=4`, and these five pin
// them to the names the rest of the tree reads them by. Both are wanted: the
// clause states the accepted *set* and would still hold if this list and
// `contract::vertex_step` drifted apart together, which is the failure these
// catch. The gap is not hypothetical — the sibling `MTLStepFunction` below is
// measured to have six of its nine names on the wrong discriminants in this very
// crate version, and `render.rs` used to express "no step function above
// PerInstance" as `> MTLVertexStepFunction::PerInstance as u32`, a band whose
// top was whatever the crate happened to say.
//
// The five ordinals come from `MTLVertexDescriptor.h` and are declared in
// `contract::vertex_step`, where the shared step/rate rule reads them. Every
// assertion here is evaluated on each arm that compiles the file, including the
// cross-compiled Metal clippy run.
const _: () = assert!(
    MTLVertexStepFunction::Constant as u32
        == crate::contract::vertex_step::MTL_VERTEX_STEP_FUNCTION_CONSTANT
);
const _: () = assert!(
    MTLVertexStepFunction::PerVertex as u32
        == crate::contract::vertex_step::MTL_VERTEX_STEP_FUNCTION_PER_VERTEX
);
const _: () = assert!(
    MTLVertexStepFunction::PerInstance as u32
        == crate::contract::vertex_step::MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE
);
const _: () = assert!(
    MTLVertexStepFunction::PerPatch as u32
        == crate::contract::vertex_step::MTL_VERTEX_STEP_FUNCTION_PER_PATCH
);
const _: () = assert!(
    MTLVertexStepFunction::PerPatchControlPoint as u32
        == crate::contract::vertex_step::MTL_VERTEX_STEP_FUNCTION_PER_PATCH_CONTROL_POINT
);

/// `MTLStepFunction` by ordinal, indexed by Apple's numbering.
///
/// This is the one conversion here that cannot name its variants, because
/// **`metal` 0.33 assigns six of this enum's nine names to the wrong numbers.**
/// Against `MTLStageInputOutputDescriptor.h` on the macOS 26 SDK:
///
/// | Apple | crate |
/// |---|---|
/// | `PerVertex` 1, `PerInstance` 2, `PerPatch` 3, `PerPatchControlPoint` 4 | 4, 1, 2, 3 |
/// | `ThreadPositionInGridY` 6, `ThreadPositionInGridXIndexed` 7 | 7, 6 |
///
/// Only `Constant` 0, `ThreadPositionInGridX` 5 and
/// `ThreadPositionInGridYIndexed` 8 agree. `stage_input.rs` already knew about
/// the second row — its own `MTL_STEP_*` constants exist for exactly that
/// reason — but its comment names only that swap, and the four vertex-side
/// values are wrong too.
///
/// So a `MTLStepFunction::PerVertex` written here would reach Apple as 4, which
/// Apple reads as `PerPatchControlPoint`: naming a variant silently rewrites
/// the guest's step function. The table is therefore indexed by Apple's
/// ordinal and holds whichever crate variant carries that discriminant, which
/// is why the entries read misaligned — they are.
///
/// The check stays exhaustive rather than a range: the crate declares 0 through
/// 8 with no gaps, so an ordinal in that run has a declared representation
/// whatever the crate calls it, and one outside it does not.
const STEP_FUNCTION_BY_ORDINAL: [MTLStepFunction; 9] = [
    MTLStepFunction::Constant,                     // 0
    MTLStepFunction::PerInstance,                  // 1  Apple: PerVertex
    MTLStepFunction::PerPatch,                     // 2  Apple: PerInstance
    MTLStepFunction::PerPatchControlPoint,         // 3  Apple: PerPatch
    MTLStepFunction::PerVertex,                    // 4  Apple: PerPatchControlPoint
    MTLStepFunction::ThreadPositionInGridX,        // 5
    MTLStepFunction::ThreadPositionInGridXIndexed, // 6  Apple: ThreadPositionInGridY
    MTLStepFunction::ThreadPositionInGridY,        // 7  Apple: ThreadPositionInGridXIndexed
    MTLStepFunction::ThreadPositionInGridYIndexed, // 8
];

/// `MTLStepFunction` for a compute stage-input buffer layout.
///
/// `const` so the pins below it are `const` assertions, checked on every arm
/// that compiles this file rather than by a suite this pathway never runs.
/// Spelled as an explicit bound rather than `get().copied()` because slice
/// indexing is not available in a `const fn`.
pub(crate) const fn step_function(ordinal: u32) -> Option<MTLStepFunction> {
    if (ordinal as usize) < STEP_FUNCTION_BY_ORDINAL.len() {
        Some(STEP_FUNCTION_BY_ORDINAL[ordinal as usize])
    } else {
        None
    }
}

checked_ordinal! {
    /// `MTLBlendFactor` for one color attachment's blend state.
    fn blend_factor -> MTLBlendFactor;
    [
        Zero, One, SourceColor, OneMinusSourceColor, SourceAlpha,
        OneMinusSourceAlpha, DestinationColor, OneMinusDestinationColor,
        DestinationAlpha, OneMinusDestinationAlpha, SourceAlphaSaturated,
        BlendColor, OneMinusBlendColor, BlendAlpha, OneMinusBlendAlpha,
        Source1Color, OneMinusSource1Color, Source1Alpha, OneMinusSource1Alpha,
    ]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLBlendOperation` for one color attachment's blend state.
    fn blend_operation -> MTLBlendOperation;
    [Add, Subtract, ReverseSubtract, Min, Max]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLCullMode` for the render encoder's raster state.
    fn cull_mode -> MTLCullMode;
    [None, Front, Back]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLWinding` for the render encoder's raster state.
    fn winding -> MTLWinding;
    [Clockwise, CounterClockwise]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLTriangleFillMode` for the render encoder's raster state.
    fn fill_mode -> MTLTriangleFillMode;
    [Fill, Lines]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLDepthClipMode` for the render encoder's raster state.
    fn depth_clip_mode -> MTLDepthClipMode;
    [Clip, Clamp]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLVisibilityResultMode` for an occlusion query armed on a draw.
    ///
    /// `Disabled` is listed because it is a declared variant and the ordinal a
    /// guest sends to disarm; the render encoder decides what to do with it.
    fn visibility_result_mode -> MTLVisibilityResultMode;
    [Disabled, Boolean, Counting]
    apple_numbers_them_from_zero;
}
// The modes this table converts are exactly the ones the device's contract says
// both backends record, plus the disarming ordinal the render encoder refuses
// by name.
//
// A `const` block rather than a `#[test]`, for the reason `checked_ordinal!`
// states and `primitive_type` below relies on: this file's tests run on Apple
// hosts only, and the cross-compiled Metal arm evaluates this one from Linux.
// The Vulkan translator carries the same pin as a test, so the two spellings of
// one Apple enum cannot drift apart on a host that can only build one of them.
const _: () = {
    use crate::contract::visibility::{
        visibility_result_mode_recordable, VISIBILITY_RESULT_MODE_DISABLED,
        VISIBILITY_RESULT_MODE_SWEEP_END,
    };
    let mut mtl = 0u32;
    while mtl < VISIBILITY_RESULT_MODE_SWEEP_END {
        let converts = visibility_result_mode(mtl).is_some();
        let contract =
            visibility_result_mode_recordable(mtl) || mtl == VISIBILITY_RESULT_MODE_DISABLED;
        assert!(
            converts == contract,
            "the Metal visibility-mode table and the device contract disagree \
             about which occlusion query modes exist",
        );
        mtl += 1;
    }
};

checked_ordinal! {
    /// `MTLCompareFunction` for a depth or stencil test.
    fn compare_function -> MTLCompareFunction;
    [Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLStencilOperation` for one stencil face.
    fn stencil_operation -> MTLStencilOperation;
    [
        Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert,
        IncrementWrap, DecrementWrap,
    ]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLLoadAction` for a render pass attachment.
    fn load_action -> MTLLoadAction;
    [DontCare, Load, Clear]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLStoreAction` for a render pass attachment.
    fn store_action -> MTLStoreAction;
    [
        DontCare, Store, MultisampleResolve, StoreAndMultisampleResolve,
        Unknown, CustomSampleDepthStore,
    ]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLIndexType` for an indexed draw or a stage-input index buffer.
    fn index_type -> MTLIndexType;
    [UInt16, UInt32]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLPrimitiveType` for a draw.
    fn primitive_type -> MTLPrimitiveType;
    [Point, Line, LineStrip, Triangle, TriangleStrip]
    apple_numbers_them_from_zero;
}
// The primitive types this rail can encode are exactly the ones the device
// tells the guest it may draw.
//
// `contract::draw::EXECUTABLE_PRIMITIVE_TYPES` leaves this device as
// device-info key 11, which the guest reads as a *permission* mask — a bit set
// there without an arm above is a draw the guest was invited to make and this
// rail then refuses. A `const` block rather than a `#[test]` for the reason
// stated inside `checked_ordinal!`: this file's tests run on Apple hosts only,
// and the cross-compiled Metal arm evaluates this one from Linux. The Vulkan
// translator carries the same pin as a test, because its arms are not `const`.
const _: () = {
    let mut mtl = 0u32;
    while mtl <= 8 {
        assert!(
            primitive_type(mtl).is_some() == crate::contract::draw::primitive_type_executable(mtl),
            "a primitive type this device advertises has no Metal arm, or vice versa",
        );
        mtl += 1;
    }
};

checked_ordinal! {
    /// `MTLSamplerMinMagFilter` for a sampler's minification or magnification.
    fn sampler_min_mag_filter -> MTLSamplerMinMagFilter;
    [Nearest, Linear]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLSamplerMipFilter` for a sampler's mip selection.
    fn sampler_mip_filter -> MTLSamplerMipFilter;
    [NotMipmapped, Nearest, Linear]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLSamplerAddressMode` for one sampler axis.
    fn sampler_address_mode -> MTLSamplerAddressMode;
    [
        ClampToEdge, MirrorClampToEdge, Repeat, MirrorRepeat, ClampToZero,
        ClampToBorderColor,
    ]
    apple_numbers_them_from_zero;
}

checked_ordinal! {
    /// `MTLSamplerBorderColor` for a sampler clamping to a border.
    fn sampler_border_color -> MTLSamplerBorderColor;
    [TransparentBlack, OpaqueBlack, OpaqueWhite]
    apple_numbers_them_from_zero;
}

/// Assert a conversion answers `ordinal` with exactly `variant`, at compile time.
///
/// `Option<MTL*>` cannot be compared with `==` in a `const` block — `metal`
/// derives `PartialEq` on only some of these enums, and none of the derives are
/// `const` — so the comparison is on the ordinal, through a `match`.
macro_rules! const_converts {
    ($fn_name:ident($ordinal:expr) == $ty:ty : $variant:ident) => {
        const _: () = assert!(match $fn_name($ordinal) {
            Some(got) => got as u32 == <$ty>::$variant as u32,
            None => false,
        });
    };
}

// The two format enums have a hole at 43 and 44, and that is the whole reason
// this module exists rather than a `<= max` bound at each call site. Apple's
// `MTLVertexDescriptor.h` and `MTLStageInputOutputDescriptor.h` on the macOS 26
// SDK run `UChar4Normalized_BGRA = 42` straight to `UChar = 45`, so a check that
// only rejects values above the last variant lets two undefined discriminants
// through.
const _: () = assert!(vertex_format(43).is_none());
const _: () = assert!(vertex_format(44).is_none());
const _: () = assert!(attribute_format(43).is_none());
const _: () = assert!(attribute_format(44).is_none());
const_converts!(vertex_format(42) == MTLVertexFormat: UChar4Normalized_BGRA);
const_converts!(vertex_format(45) == MTLVertexFormat: UChar);
const_converts!(attribute_format(42) == MTLAttributeFormat: UChar4Normalized_BGRA);
const_converts!(attribute_format(45) == MTLAttributeFormat: UChar);

// The four values the SDK declares and `metal` does not, pinned so the narrowing
// this module's doc describes stays a measured fact. If a `metal` bump adds them
// these assertions flip, which is the signal to add the variants above and
// delete these lines rather than to relax them.
//
// MTLVertexFormatFloatRG11B10 / MTLVertexFormatFloatRGB9E5.
const _: () = assert!(vertex_format(54).is_none());
const _: () = assert!(vertex_format(55).is_none());
const _: () = assert!(attribute_format(54).is_none());
const _: () = assert!(attribute_format(55).is_none());
// MTLBlendFactorUnspecialized / MTLBlendOperationUnspecialized.
const _: () = assert!(blend_factor(19).is_none());
const _: () = assert!(blend_operation(5).is_none());

// The step-function table carries Apple's numbering, and every entry's own
// discriminant is its index.
//
// This is what pins `metal` 0.33's misnumbering rather than working around it
// silently: if a crate bump renumbers the variants to match Apple, the table
// above still produces the right ordinals and this still holds, but the
// misaligned-looking comments become wrong. If a bump renumbers them some
// *other* way, the build fails here.
const _: () = {
    let mut ordinal = 0usize;
    while ordinal < STEP_FUNCTION_BY_ORDINAL.len() {
        assert!(STEP_FUNCTION_BY_ORDINAL[ordinal] as u32 == ordinal as u32);
        assert!(match step_function(ordinal as u32) {
            Some(got) => got as u32 == ordinal as u32,
            None => false,
        });
        ordinal += 1;
    }
};
const _: () = assert!(step_function(STEP_FUNCTION_BY_ORDINAL.len() as u32).is_none());
const _: () = assert!(step_function(u32::MAX).is_none());

// The names `metal` 0.33 gives the step function are not Apple's, and the three
// that happen to agree are the only ones that may be used by name.
//
// Apple: `PerVertex` 1, `PerInstance` 2, `PerPatch` 3, `PerPatchControlPoint` 4,
// `ThreadPositionInGridY` 6, `ThreadPositionInGridXIndexed` 7.
const _: () = assert!(MTLStepFunction::Constant as u32 == 0);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridX as u32 == 5);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridYIndexed as u32 == 8);
// Everything else is off, so naming it would rewrite the guest's value.
const _: () = assert!(MTLStepFunction::PerVertex as u32 != 1);
const _: () = assert!(MTLStepFunction::PerInstance as u32 != 2);
const _: () = assert!(MTLStepFunction::PerPatch as u32 != 3);
const _: () = assert!(MTLStepFunction::PerPatchControlPoint as u32 != 4);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridY as u32 != 6);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridXIndexed as u32 != 7);

checked_ordinal! {
    /// `MTLPixelFormat` for a colour, depth or stencil attachment.
    ///
    /// The last member of this class, and the one that stayed a `transmute`
    /// longest: this enum is by far the sparsest here — 139 declared values
    /// scattered over `0..=555` — so "not above the last variant" admits
    /// hundreds of codes that name nothing, and an undeclared one is undefined
    /// behaviour rather than a format Metal will reject. Guest attachment
    /// records reach it directly, so the ordinal really is arbitrary.
    ///
    /// No `apple_numbers_them_from_zero` clause: the numbering is not
    /// contiguous and is not meant to be. The compressed families sit in blocks
    /// Apple spaces apart, and the depth/stencil group starts at 250 with
    /// `X32_Stencil8` and `X24_Stencil8` above the pair they alias.
    ///
    /// This lists everything `metal` 0.33 declares rather than the subset
    /// `crate::contract::pixel_format` sizes. The two are different questions —
    /// this one is "can the Rust type hold this value at all", and answering it
    /// with the narrower table would refuse formats this device hands to Metal
    /// unexamined today. What each call site does with a format it cannot size
    /// stays that call site's decision.
    fn pixel_format -> MTLPixelFormat;
    [
        Invalid, A8Unorm, R8Unorm, R8Unorm_sRGB, R8Snorm, R8Uint, R8Sint, R16Unorm,
        R16Snorm, R16Uint, R16Sint, R16Float, RG8Unorm, RG8Unorm_sRGB, RG8Snorm,
        RG8Uint, RG8Sint, B5G6R5Unorm, A1BGR5Unorm, ABGR4Unorm, BGR5A1Unorm, R32Uint,
        R32Sint, R32Float, RG16Unorm, RG16Snorm, RG16Uint, RG16Sint, RG16Float,
        RGBA8Unorm, RGBA8Unorm_sRGB, RGBA8Snorm, RGBA8Uint, RGBA8Sint, BGRA8Unorm,
        BGRA8Unorm_sRGB, RGB10A2Unorm, RGB10A2Uint, RG11B10Float, RGB9E5Float,
        BGR10A2Unorm, RG32Uint, RG32Sint, RG32Float, RGBA16Unorm, RGBA16Snorm,
        RGBA16Uint, RGBA16Sint, RGBA16Float, RGBA32Uint, RGBA32Sint, RGBA32Float,
        BC1_RGBA, BC1_RGBA_sRGB, BC2_RGBA, BC2_RGBA_sRGB, BC3_RGBA, BC3_RGBA_sRGB,
        BC4_RUnorm, BC4_RSnorm, BC5_RGUnorm, BC5_RGSnorm, BC6H_RGBFloat,
        BC6H_RGBUfloat, BC7_RGBAUnorm, BC7_RGBAUnorm_sRGB, PVRTC_RGB_2BPP,
        PVRTC_RGB_2BPP_sRGB, PVRTC_RGB_4BPP, PVRTC_RGB_4BPP_sRGB, PVRTC_RGBA_2BPP,
        PVRTC_RGBA_2BPP_sRGB, PVRTC_RGBA_4BPP, PVRTC_RGBA_4BPP_sRGB, EAC_R11Unorm,
        EAC_R11Snorm, EAC_RG11Unorm, EAC_RG11Snorm, EAC_RGBA8, EAC_RGBA8_sRGB,
        ETC2_RGB8, ETC2_RGB8_sRGB, ETC2_RGB8A1, ETC2_RGB8A1_sRGB, ASTC_4x4_sRGB,
        ASTC_5x4_sRGB, ASTC_5x5_sRGB, ASTC_6x5_sRGB, ASTC_6x6_sRGB, ASTC_8x5_sRGB,
        ASTC_8x6_sRGB, ASTC_8x8_sRGB, ASTC_10x5_sRGB, ASTC_10x6_sRGB, ASTC_10x8_sRGB,
        ASTC_10x10_sRGB, ASTC_12x10_sRGB, ASTC_12x12_sRGB, ASTC_4x4_LDR, ASTC_5x4_LDR,
        ASTC_5x5_LDR, ASTC_6x5_LDR, ASTC_6x6_LDR, ASTC_8x5_LDR, ASTC_8x6_LDR,
        ASTC_8x8_LDR, ASTC_10x5_LDR, ASTC_10x6_LDR, ASTC_10x8_LDR, ASTC_10x10_LDR,
        ASTC_12x10_LDR, ASTC_12x12_LDR, ASTC_4x4_HDR, ASTC_5x4_HDR, ASTC_5x5_HDR,
        ASTC_6x5_HDR, ASTC_6x6_HDR, ASTC_8x5_HDR, ASTC_8x6_HDR, ASTC_8x8_HDR,
        ASTC_10x5_HDR, ASTC_10x6_HDR, ASTC_10x8_HDR, ASTC_10x10_HDR, ASTC_12x10_HDR,
        ASTC_12x12_HDR, GBGR422, BGRG422, Depth16Unorm, Depth32Float, Stencil8,
        Depth24Unorm_Stencil8, Depth32Float_Stencil8, X32_Stencil8, X24_Stencil8,
        BGRA10_XR, BGRA10_XR_SRGB, BGR10_XR, BGR10_XR_SRGB,
    ]
}

// `pixel_format` cannot carry `apple_numbers_them_from_zero` — its numbering is
// sparse by design — so the accepted set is otherwise whatever `metal` 0.33
// assigns, and a crate bump that renumbered it would move the set with every
// assertion above still green. That is exactly the hole the clause closes for
// the contiguous tables. These pin the same thing for this one: the anchor of
// each block, read off `MTLPixelFormat.h`. They are claims about Apple's
// numbers that the compiler checks against `metal`'s.
const _: () = {
    assert!(MTLPixelFormat::Invalid as u32 == 0);
    assert!(MTLPixelFormat::A8Unorm as u32 == 1);
    // The uncompressed run, which is what this device actually serves.
    assert!(MTLPixelFormat::R8Unorm as u32 == 10);
    assert!(MTLPixelFormat::RGBA8Unorm as u32 == 70);
    assert!(MTLPixelFormat::BGRA8Unorm as u32 == 80);
    assert!(MTLPixelFormat::RGBA32Float as u32 == 125);
    // First compressed block, and the YUV pair that ends them.
    assert!(MTLPixelFormat::BC1_RGBA as u32 == 130);
    assert!(MTLPixelFormat::GBGR422 as u32 == 240);
    // Depth/stencil. `X32_Stencil8` and `X24_Stencil8` sit *above* the pair they
    // alias, which is why this group is listed in declaration order rather than
    // by aspect.
    assert!(MTLPixelFormat::Depth16Unorm as u32 == 250);
    assert!(MTLPixelFormat::Depth32Float as u32 == 252);
    assert!(MTLPixelFormat::Stencil8 as u32 == 253);
    assert!(MTLPixelFormat::Depth24Unorm_Stencil8 as u32 == 255);
    assert!(MTLPixelFormat::Depth32Float_Stencil8 as u32 == 260);
    assert!(MTLPixelFormat::X32_Stencil8 as u32 == 261);
    assert!(MTLPixelFormat::X24_Stencil8 as u32 == 262);
    // The tail the const sweep's ceiling is measured from.
    assert!(MTLPixelFormat::BGR10_XR_SRGB as u32 == 555);
};
