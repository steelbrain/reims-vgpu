//! Typed decline reasons for the Metal → Vulkan translation boundary.
//!
//! Every translation entry point is total: it returns either a Vulkan value or
//! one of these, never a silent default. The variants exist so a decline can be
//! logged, tested and grepped by **name** — `AGENTS.md` requires each distinct
//! check to carry its own `reason=<slug>` rather than collapsing several causes
//! into one status. A free-text payload cannot satisfy that mechanically.
//!
//! Shape is a plain enum implementing [`Decline`] plus
//! [`TranslateReason::slug`]. The offending numeric value
//! rides along so the fail-visible line carries the load-bearing field, and
//! [`std::fmt::Display`] renders both.
//!
//! # Two classes of variant, and only one of them is this device's fault
//!
//! Several variants below already say they are "distinct from
//! `UnknownPixelFormat`" — the format is understood, this rail just does not
//! carry it. That distinction is the whole taxonomy, and it decides what the
//! repair is:
//!
//! - **The value is out of contract.** `Unknown*` — an ordinal Apple's
//!   serializer never emits. A real device rejects it too, so the refusal *is*
//!   the correct behaviour and there is nothing to implement.
//! - **The value is in contract and this backend has no path for it.**
//!   `NoSampledLayout`, `NoColorAttachmentFormat`, `NoStorageImageFormat`,
//!   `VertexStepFunctionPerPatch`, `FormatNotVertexBuffer`. The guest asked for
//!   something legal and lost the work. These are gaps, and the repair is to
//!   build the path — never to stop advertising the capability, unless *no host*
//!   could ever serve it. `reims-vgpu::model::DEVICE_INFO_KEY_FRAMEBUFFER_READ`
//!   carries that rule and the pair of cases that establish it.
//! - **The value is in contract, this backend has a path, and the path would
//!   answer something else.** `VertexFormatWidenReadAsFour`,
//!   `VertexFormatWidenShaderUnreadable` — the only two, and both belong to the
//!   vertex-format widening fallback in [`super::support`]. What makes them a
//!   class of their own is that the alternative to refusing is not losing the
//!   work, it is *executing a different command*: the substitute format binds
//!   fine and hands the shader a fourth component the guest's own format would
//!   have defaulted to 1.0. Refusing is the faithful answer, and the repair is
//!   not to widen the fallback but to narrow when it applies.
//!
//! **`NoSampledLayout` fires in the hundreds of thousands, and the zero that
//! used to be recorded here was an artifact of where it was sampled.** The
//! linear sampled zero-copy rung matched `Err(_)` and threw the reason away,
//! emitting a route name of its own instead, so the slug reached no fail line
//! however often the check refused — 22 270 declines in one driven macos-26
//! boot, against 8 975 on macos-13. The reading "this class has never fired"
//! was taken from the fail log and the fail log could not see it. Any future
//! claim of that shape has to name the site that would emit, not just the
//! absence of a grep hit.
//!
//! There was briefly a fourth pixel-format variant here,
//! `SampledComponentsNotIdentity`, for a format whose Metal channels need a
//! swizzle to sit on their Vulkan ones. It existed to size that class against
//! the others, it did — `A8Unorm`, 9 198 declines a boot on macos-13 — and the
//! repair it pointed at then removed the decline entirely, because
//! `sampled_pixels` now hands the format's plan back instead of refusing over
//! it. It is gone rather than kept as a healthy zero: nothing can construct it,
//! and a reason with no producer is a claim the taxonomy cannot keep.
//!
//! **The rest of the second class, and all of the third, still read zero on
//! archived boots of this rig** — x86/PCI/Vulkan, driven under the window-drag
//! and web-content probes as well as idle. For the second, each names a real
//! guest feature this workload does not reach, which is the reading that makes
//! leaving them open a measurement rather than a bet. For the third the zero is
//! structural on this
//! rig and says less: no host here declines a three-component vertex format, so
//! the fallback is never entered and neither variant is reachable. It says
//! nothing about the arm64 pathway, which this checkout cannot boot, and a
//! firing is the signal that one has become worth building.

use reims_vgpu_observe::Decline;

/// Why a decoded Metal value has no Vulkan equivalent this backend will emit.
///
/// The payload is always the raw wire value the decoder produced, so a log line
/// names both the class of failure and the exact number that caused it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslateReason {
    /// `MTLPixelFormat` value outside the set the decode contract defines.
    /// "Unknown wire format stays unknown" — no fallback texel layout is
    /// invented for it.
    UnknownPixelFormat(u16),
    /// A pipeline honoured the format's channel layout but **not** its sRGB
    /// transfer function, so the hardware will not encode on write. Recorded by
    /// the site that takes [`super::pixel::PixelFormat::linear_vk`] instead of
    /// `vk`; it is a downgrade, not a failure, and the draw still runs.
    SrgbDowngraded(u16),
    /// The format is defined by the contract but the sampled rail carries no
    /// byte layout for it, so its texels cannot be uploaded without a CPU
    /// convert pass. Distinct from [`Self::UnknownPixelFormat`]: the format is
    /// understood, this rail just does not carry it.
    NoSampledLayout(u16),
    /// The format is defined by the contract but is not one the engine can use
    /// as a colour attachment. Distinct from [`Self::UnknownPixelFormat`] for
    /// the same reason.
    NoColorAttachmentFormat(u16),
    /// The format is defined by the contract but the compute rail carries no
    /// storage-image layout for it. Same shape as [`Self::NoSampledLayout`]:
    /// the format is understood, this rail just does not carry it.
    NoStorageImageFormat(u16),
    /// `MTLVertexFormat` value outside the SDK enum.
    UnknownVertexFormat(u32),
    /// `MTLVertexStepFunction` value outside the SDK enum.
    ///
    /// The SDK enum runs 0-4, not 0-2: 3 and 4 are `PerPatch` and
    /// `PerPatchControlPoint`, and they get [`Self::VertexStepFunctionPerPatch`]
    /// rather than this. See that variant for why the distinction is worth a
    /// second slug.
    UnknownVertexStepFunction(u32),
    /// `MTLVertexStepFunctionPerPatch` (3) or `PerPatchControlPoint` (4) — the
    /// two tessellation step rates.
    ///
    /// Both used to land on [`Self::UnknownVertexStepFunction`], which says this
    /// device did not recognise the value. It does: these are declared SDK
    /// values with a known meaning, and what is missing is a *Vulkan* spelling.
    /// `VkVertexInputRate` has only `VERTEX` and `INSTANCE`; per-patch input
    /// rates belong to a tessellation pipeline this backend does not build, so
    /// the attribute is genuinely unbindable here — but for a reason a reader
    /// can act on.
    ///
    /// The difference is not cosmetic. `unknown_vertex_step_function step=3` in
    /// a driven boot's log sends the next reader looking for a decode bug that
    /// does not exist; `vertex_step_function_per_patch` says the guest ran a
    /// tessellation pipeline and names what would have to be built. The Metal
    /// arm does carry these — `runtime::icb` binds `PerPatchControlPoint` for
    /// post-tessellation vertex functions — so this is an arm difference rather
    /// than a device-wide gap, which is exactly the kind of thing one shared
    /// slug hides.
    VertexStepFunctionPerPatch(u32),
    /// `MTLPrimitiveType` value outside the SDK enum.
    UnknownPrimitiveType(u32),
    /// `MTLBlendFactor` value outside the SDK enum.
    UnknownBlendFactor(u32),
    /// `MTLBlendOperation` value outside the SDK enum.
    UnknownBlendOperation(u32),
    /// `MTLCompareFunction` value outside the SDK enum (depth, stencil and
    /// sampler compare share the Metal enum, hence one reason).
    UnknownCompareFunction(u32),
    /// `MTLStencilOperation` value outside the SDK enum.
    UnknownStencilOperation(u32),
    /// `MTLCullMode` value outside the SDK enum.
    UnknownCullMode(u32),
    /// `MTLWinding` value outside the SDK enum.
    UnknownWinding(u32),
    /// `MTLTriangleFillMode` value outside the SDK enum.
    UnknownFillMode(u32),
    /// `MTLDepthClipMode` value outside the SDK enum.
    UnknownDepthClipMode(u32),
    /// `MTLSamplerMinMagFilter` value outside the SDK enum.
    UnknownSamplerFilter(u32),
    /// `MTLSamplerMipFilter` value outside the SDK enum.
    UnknownSamplerMipFilter(u32),
    /// `MTLSamplerAddressMode` value outside the SDK enum.
    UnknownSamplerAddressMode(u32),
    /// `MTLSamplerBorderColor` value outside the SDK enum.
    UnknownSamplerBorderColor(u32),
    /// A type-8 view swizzle selector outside the decoded contract's range.
    UnknownSwizzleSelector(u8),
    /// The device does not advertise the requested `VkFormat` as a vertex
    /// buffer format, and no portable substitute exists for it.
    /// Payload is the `VkFormat` raw value.
    FormatNotVertexBuffer(i32),
    /// The device declined a three-component vertex format, its four-component
    /// substitute would fit, **and the shader reads all four components** — so
    /// binding the substitute would supply a fourth component out of the vertex
    /// buffer where the guest's own format defaults it to 1.0.
    ///
    /// Distinct from [`Self::FormatNotVertexBuffer`]: there a substitute does
    /// not exist or does not fit, here it exists and fits and is still the
    /// wrong answer. Payload is the `VkFormat` raw value the guest asked for.
    VertexFormatWidenReadAsFour(i32),
    /// The same substitution, declined because the shader's declared width
    /// could not be read: an input type that is not a scalar or a vector, or a
    /// module whose instruction stream did not parse.
    ///
    /// Its own slug because the repair is different. A firing of
    /// [`Self::VertexFormatWidenReadAsFour`] is this device correctly refusing
    /// something it cannot represent; a firing of this one means
    /// [`crate::spirv_vertex_input`] met a module shape it does not
    /// handle, and the repair is to teach it that shape.
    VertexFormatWidenShaderUnreadable(i32),
}

impl reims_vgpu_observe::Decline for TranslateReason {
    /// Stable snake_case slug for `reason=` in the always-on fail log.
    ///
    /// One slug per distinct check, never shared: the point is that a grep of
    /// `/tmp/reims-vgpu-fail.log` tells you which translation refused, not merely that
    /// one did.
    ///
    /// This was an inherent method with a per-enum uniqueness test, which is how
    /// `unknown_pixel_format` came to be claimed by `runtime/heap_query`'s
    /// `QueryError` as well: both enums were internally consistent and nothing
    /// compared them. Implementing the crate trait gives every slug here one
    /// vocabulary to be distinct within.
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownPixelFormat(_) => "unknown_pixel_format",
            Self::NoStorageImageFormat(_) => "no_storage_image_format",
            Self::SrgbDowngraded(_) => "srgb_downgraded",
            Self::NoSampledLayout(_) => "no_sampled_layout",
            Self::NoColorAttachmentFormat(_) => "no_color_attachment_format",
            Self::UnknownVertexFormat(_) => "unknown_vertex_format",
            Self::UnknownVertexStepFunction(_) => "unknown_vertex_step_function",
            Self::VertexStepFunctionPerPatch(_) => "vertex_step_function_per_patch",
            Self::UnknownPrimitiveType(_) => "unknown_primitive_type",
            Self::UnknownBlendFactor(_) => "unknown_blend_factor",
            Self::UnknownBlendOperation(_) => "unknown_blend_operation",
            Self::UnknownCompareFunction(_) => "unknown_compare_function",
            Self::UnknownStencilOperation(_) => "unknown_stencil_operation",
            Self::UnknownCullMode(_) => "unknown_cull_mode",
            Self::UnknownWinding(_) => "unknown_winding",
            Self::UnknownFillMode(_) => "unknown_fill_mode",
            Self::UnknownDepthClipMode(_) => "unknown_depth_clip_mode",
            Self::UnknownSamplerFilter(_) => "unknown_sampler_filter",
            Self::UnknownSamplerMipFilter(_) => "unknown_sampler_mip_filter",
            Self::UnknownSamplerAddressMode(_) => "unknown_sampler_address_mode",
            Self::UnknownSamplerBorderColor(_) => "unknown_sampler_border_color",
            Self::UnknownSwizzleSelector(_) => "unknown_swizzle_selector",
            Self::FormatNotVertexBuffer(_) => "format_not_vertex_buffer",
            Self::VertexFormatWidenReadAsFour(_) => "vertex_format_widen_read_as_four",
            Self::VertexFormatWidenShaderUnreadable(_) => "vertex_format_widen_shader_unreadable",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("value", self.value().to_string())]
    }
}

impl TranslateReason {
    /// The raw decoded value that could not be translated, widened to `u32` for
    /// uniform logging. `VkFormat` values are `i32` on the wire and are
    /// reinterpreted, not truncated.
    pub fn value(self) -> u32 {
        match self {
            Self::UnknownPixelFormat(v)
            | Self::SrgbDowngraded(v)
            | Self::NoSampledLayout(v)
            | Self::NoColorAttachmentFormat(v)
            | Self::NoStorageImageFormat(v) => u32::from(v),
            Self::UnknownVertexFormat(v)
            | Self::UnknownVertexStepFunction(v)
            | Self::VertexStepFunctionPerPatch(v)
            | Self::UnknownPrimitiveType(v)
            | Self::UnknownBlendFactor(v)
            | Self::UnknownBlendOperation(v)
            | Self::UnknownCompareFunction(v)
            | Self::UnknownStencilOperation(v)
            | Self::UnknownCullMode(v)
            | Self::UnknownWinding(v)
            | Self::UnknownFillMode(v)
            | Self::UnknownDepthClipMode(v)
            | Self::UnknownSamplerFilter(v)
            | Self::UnknownSamplerMipFilter(v)
            | Self::UnknownSamplerAddressMode(v)
            | Self::UnknownSamplerBorderColor(v) => v,
            Self::UnknownSwizzleSelector(v) => u32::from(v),
            Self::FormatNotVertexBuffer(v)
            | Self::VertexFormatWidenReadAsFour(v)
            | Self::VertexFormatWidenShaderUnreadable(v) => v as u32,
        }
    }
}

impl std::fmt::Display for TranslateReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={} value={}", self.slug(), self.value())
    }
}

impl std::error::Error for TranslateReason {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reason this module can produce, so the exhaustiveness tests below
    /// fail to compile-or-assert when a variant is added without a slug.
    ///
    /// Kept honest by [`every_variant_is_in_the_all_list_exactly_once`]. Until
    /// that test existed this list was hand-written with nothing checking it,
    /// so the "fail to compile" half of the sentence above was not true — a new
    /// variant simply went unswept, which is how `VertexStepFunctionPerPatch`
    /// was added and every test here still passed.
    const ALL: &[TranslateReason] = &[
        TranslateReason::UnknownPixelFormat(0),
        TranslateReason::SrgbDowngraded(0),
        TranslateReason::NoSampledLayout(0),
        TranslateReason::NoColorAttachmentFormat(0),
        TranslateReason::NoStorageImageFormat(0),
        TranslateReason::UnknownVertexFormat(0),
        TranslateReason::UnknownVertexStepFunction(0),
        TranslateReason::VertexStepFunctionPerPatch(0),
        TranslateReason::UnknownPrimitiveType(0),
        TranslateReason::UnknownBlendFactor(0),
        TranslateReason::UnknownBlendOperation(0),
        TranslateReason::UnknownCompareFunction(0),
        TranslateReason::UnknownStencilOperation(0),
        TranslateReason::UnknownCullMode(0),
        TranslateReason::UnknownWinding(0),
        TranslateReason::UnknownFillMode(0),
        TranslateReason::UnknownDepthClipMode(0),
        TranslateReason::UnknownSamplerFilter(0),
        TranslateReason::UnknownSamplerMipFilter(0),
        TranslateReason::UnknownSamplerAddressMode(0),
        TranslateReason::UnknownSamplerBorderColor(0),
        TranslateReason::UnknownSwizzleSelector(0),
        TranslateReason::FormatNotVertexBuffer(0),
        TranslateReason::VertexFormatWidenReadAsFour(0),
        TranslateReason::VertexFormatWidenShaderUnreadable(0),
    ];

    /// [`ALL`] really does hold every variant, exactly once.
    ///
    /// The list's own doc claimed the tests below "fail to compile-or-assert
    /// when a variant is added", and only the assert half was true: `ALL` is a
    /// hand-written array, so a new variant was simply swept one fewer time and
    /// every test still passed. `VertexStepFunctionPerPatch` was added that way.
    ///
    /// The match below has no wildcard, so a new variant fails to compile here.
    /// Tying the set of distinct arms to `ALL.len()` closes the other
    /// direction: adding the arm without adding the entry fails too. Between
    /// them, the sentence on `ALL` is now a check rather than a claim.
    #[test]
    fn every_variant_is_in_the_all_list_exactly_once() {
        fn index(reason: TranslateReason) -> usize {
            match reason {
                TranslateReason::UnknownPixelFormat(_) => 0,
                TranslateReason::SrgbDowngraded(_) => 1,
                TranslateReason::NoSampledLayout(_) => 2,
                TranslateReason::NoColorAttachmentFormat(_) => 3,
                TranslateReason::NoStorageImageFormat(_) => 4,
                TranslateReason::UnknownVertexFormat(_) => 5,
                TranslateReason::UnknownVertexStepFunction(_) => 6,
                TranslateReason::VertexStepFunctionPerPatch(_) => 7,
                TranslateReason::UnknownPrimitiveType(_) => 8,
                TranslateReason::UnknownBlendFactor(_) => 9,
                TranslateReason::UnknownBlendOperation(_) => 10,
                TranslateReason::UnknownCompareFunction(_) => 11,
                TranslateReason::UnknownStencilOperation(_) => 12,
                TranslateReason::UnknownCullMode(_) => 13,
                TranslateReason::UnknownWinding(_) => 14,
                TranslateReason::UnknownFillMode(_) => 15,
                TranslateReason::UnknownDepthClipMode(_) => 16,
                TranslateReason::UnknownSamplerFilter(_) => 17,
                TranslateReason::UnknownSamplerMipFilter(_) => 18,
                TranslateReason::UnknownSamplerAddressMode(_) => 19,
                TranslateReason::UnknownSamplerBorderColor(_) => 20,
                TranslateReason::UnknownSwizzleSelector(_) => 21,
                TranslateReason::FormatNotVertexBuffer(_) => 22,
                TranslateReason::VertexFormatWidenReadAsFour(_) => 23,
                TranslateReason::VertexFormatWidenShaderUnreadable(_) => 24,
            }
        }
        let mut seen: Vec<usize> = ALL.iter().map(|r| index(*r)).collect();
        seen.sort_unstable();
        let listed = seen.len();
        seen.dedup();
        assert_eq!(
            listed,
            seen.len(),
            "a variant appears in ALL more than once"
        );
        assert_eq!(
            seen,
            (0..ALL.len()).collect::<Vec<_>>(),
            "ALL is missing a variant, or holds one the match does not name"
        );
    }

    /// Two checks sharing a slug is the exact failure `AGENTS.md` names: you
    /// grep the fail log, see the slug fire, and still cannot tell which of the
    /// two refused.
    #[test]
    fn every_reason_has_its_own_slug() {
        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate translate reason slug");
    }

    /// Slugs are grepped out of a space-separated log line, so they may not
    /// carry whitespace or an `=`, and they stay kebab/snake for consistency
    /// with the existing `caps` and `present_proxy` slugs.
    #[test]
    fn slugs_are_log_safe() {
        for r in ALL {
            let s = r.slug();
            assert!(!s.is_empty());
            assert!(
                s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "slug {s:?} must be lowercase snake_case"
            );
        }
    }

    /// The payload survives into the rendered line — a decline that names only
    /// its class leaves the reader without the value that caused it.
    #[test]
    fn display_carries_the_offending_value() {
        let r = TranslateReason::UnknownPixelFormat(0x1234);
        assert_eq!(r.to_string(), "reason=unknown_pixel_format value=4660");
        // VkFormat is a signed handle; reinterpretation must not truncate.
        let f = TranslateReason::FormatNotVertexBuffer(-9);
        assert_eq!(f.value(), u32::MAX - 8);
    }
}
