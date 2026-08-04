//! Typed reasons the engine declines a request it understood.
//!
//! # Why this is not a `String`
//!
//! `AGENTS.md` requires every rejected guest command to name a `reason=<slug>`
//! at the failing site, and calls out this exact shape: *"When N distinct checks
//! share one status (`Unsupported`, `MissingBuffer`, …), each needs its own slug
//! so you can tell which fired."* With free text that rule cannot be satisfied
//! mechanically or under test — you can only read the sites and hope. Twenty-odd
//! unrelated causes were sharing `DrawError::Unsupported(String)`, several of
//! them prose that differed only in wording between neighbouring branches.
//!
//! An enum makes the rule checkable: a unit test asserts no two variants share a
//! slug, and a new decline that forgets one does not compile.
//!
//! # `Unsupported` means *this device or this engine cannot*, not *the guest is
//! wrong*
//!
//! A malformed request is a validation or preparation decline; a failed Vulkan
//! call is `DrawError::VkCall`. These are the cases where the request made sense
//! and the answer is still no — which is why several of them name a capability
//! the host GPU lacks. Those are the ones that matter on the matrix rows nobody
//! here owns.

use crate::backend::vulkan::translate::TranslateReason;
use crate::observe::Decline;

/// A request the engine understood and declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReason {
    /// More than one viewport/scissor in a draw. Metal's multi-viewport
    /// rasterization is not modelled.
    /// A resident target bound as a sampled image must be a plain 2D image;
    /// arrayed and volume residents have no bind path.
    ResidentSampledNot2d { binding: u32 },
    /// Same for a zero-copy guest-run sampled bind.
    GuestRunSampledNot2d { binding: u32 },
    /// More MRT secondary attachments than the render pass can carry.
    SecondaryAttachmentCap { requested: usize, cap: usize },
    /// The translated module is an unstructured state machine, which this
    /// host's shader compiler cannot compile in bounded time.
    ///
    /// metal2vulkan structures control flow when it can and falls back to a
    /// relooper state machine when it cannot: one function, one loop, and one
    /// `OpSwitch` whose case count is the block count, with the next block index
    /// written to a variable each iteration. Measured on an NVIDIA host, one
    /// such module — the WindowServer compositor, 2 731 blocks, 2 725 cases —
    /// held `vkCreateGraphicsPipelines` for over 22 minutes at a full core with
    /// a flat working set, and never returned. The same driver compiles every
    /// structured module in that boot in single-digit milliseconds.
    ///
    /// Declining costs this shader's draws. Not declining costs the device: the
    /// call runs on the drain worker under the device lock, so the guest's rings
    /// stop being consumed and it reports a GPU hang.
    UnstructuredStateMachineShader { blocks: u32, switch_cases: u32 },
    /// The translated module is not valid SPIR-V and must not reach the driver.
    ///
    /// Specifically: an `OpCompositeInsert` or `OpCompositeExtract` moves an
    /// image or sampler handle through a composite, which the Logical addressing
    /// model cannot represent. `spirv-val` rejects it, and a driver given an
    /// invalid module is free to do anything — on the x86 rail it stopped
    /// serving the guest entirely, with no other diagnostic anywhere.
    ///
    /// This is a translator defect, not a guest one. Declining names it instead
    /// of letting the process die silently.
    InvalidTranslatedModule { pipeline_ref: u32 },
    /// The fragment shader declares a descriptor the draw never bound.
    ///
    /// Reading an undefined descriptor is undefined behaviour. The engine builds
    /// its descriptor layout purely from provided resources, so such a draw both
    /// samples whatever memory the descriptor happens to address and omits the
    /// binding from the pipeline layout — reported by a validation layer as
    /// `VUID-vkCmdDraw-None-08114` plus
    /// `VUID-VkGraphicsPipelineCreateInfo-layout-07988` on the same binding.
    ///
    /// Measured on the x86 rail: windows dragged their previous contents behind
    /// them and the process eventually faulted. The unbound indices themselves
    /// are named by the `shader_resource_declared_unbound` line emitted with
    /// this decline.
    FragmentDescriptorUnbound { pipeline_ref: u32 },
    /// The device does not advertise `samplerAnisotropy` and the guest sampler
    /// asked for it.
    SamplerAnisotropyUnsupported,
    /// The guest sampler uses `MTLSamplerAddressModeMirrorClampToEdge` and this
    /// device offers neither the Vulkan 1.2 `samplerMirrorClampToEdge` feature
    /// nor `VK_KHR_sampler_mirror_clamp_to_edge`.
    ///
    /// Binding it anyway is what this crate used to do — the translation table
    /// emitted `MIRROR_CLAMP_TO_EDGE` and nothing ever requested the feature,
    /// so the sampler was created with a mode the device had not been asked
    /// for. That is undefined behaviour a validation layer catches on someone
    /// else's GPU; declining by name is the honest answer.
    SamplerMirrorClampToEdgeUnsupported,
    /// The guest pipeline names one of `MTLBlendFactor`'s four dual-source
    /// factors (`Source1Color` .. `OneMinusSource1Alpha`, 15-18) and this device
    /// does not advertise `VkPhysicalDeviceFeatures::dualSrcBlend`.
    ///
    /// These reached no arm at all until the translation table was extended —
    /// `translate::blend::factor` stopped at 14 and its test asserted 15 was
    /// past the end of `MTLBlendFactor`, which runs to 18. So a guest asking
    /// for dual-source blending was refused as an unknown factor on every host,
    /// including the ones that support it. Now it translates, and only a host
    /// that genuinely cannot run it declines — here, by name.
    DualSourceBlendUnsupported,
    /// The device declines this vertex attribute format and no portable
    /// substitute fits. Carries the translation-layer reason so the two log
    /// lines agree on why.
    VertexFormat(TranslateReason),
    /// A constant-rate vertex attribute (`divisor == 0`) on a device without
    /// `vertexAttributeInstanceRateZeroDivisor`.
    ConstantVertexAttribute,
    /// A per-instance step rate above 1 on a device without
    /// `vertexAttributeInstanceRateDivisor`.
    InstanceRateDivisorUnsupported { step_rate: u32 },
    /// A per-instance step rate above the device's `maxVertexAttribDivisor`.
    InstanceRateDivisorOverLimit { step_rate: u32, limit: u32 },
    /// No queue family supports graphics and compute together, which the
    /// engine's single-queue submit model requires.
    NoCombinedGraphicsComputeQueue,
    // The memory-type lookups. Each is a `memory_type_for(bits, class)` that
    // found nothing: the device advertises no memory type satisfying the buffer
    // or image's requirement bits under the class this allocation needs. That is
    // a device *capability* refusal, not a failed Vulkan call — it matters on the
    // matrix rows nobody here owns, where a class an NVIDIA host offers may be
    // absent. Named per purpose because "which allocation had nowhere to live" is
    // the diagnostic; each carries the requirement bits that matched no type.
    /// No host-visible memory type for a staging (upload) buffer.
    NoHostVisibleMemoryForStaging { memory_type_bits: u32 },
    /// No host-visible memory type for a readback buffer.
    NoHostVisibleMemoryForReadback { memory_type_bits: u32 },
    /// No host-visible memory type for the stats-reduction readback buffer.
    NoHostVisibleMemoryForStats { memory_type_bits: u32 },
    /// No device-local memory type for a storage image.
    NoDeviceLocalMemoryForStorageImage { memory_type_bits: u32 },
    /// No device-local memory type for a shared optimal-image slab.
    NoDeviceLocalMemoryForSlab { memory_type_bits: u32 },
    /// No device-local memory type for an MRT secondary attachment image.
    NoDeviceLocalMemoryForMrtSecondary { memory_type_bits: u32 },
    /// No device-local memory type for a depth attachment image.
    NoDeviceLocalMemoryForDepth { memory_type_bits: u32 },
    /// `VK_KHR_swapchain` is not enabled on the engine device.
    SwapchainUnavailable,
    /// The engine's queue family cannot present to the host window's surface.
    QueueCannotPresent { queue_family: u32 },
    /// The surface's swapchain images cannot be a transfer destination, which
    /// the present blit requires.
    SwapchainLacksTransferDst,
    /// The surface advertises no formats at all.
    SwapchainNoSurfaceFormat,
    /// The surface advertises no composite-alpha mode.
    SwapchainNoCompositeAlpha,
}

impl crate::observe::Decline for DrawReason {
    /// Stable slug for `reason=` in the always-on fail log. One per distinct
    /// check, never shared.
    fn slug(&self) -> &'static str {
        match self {
            Self::ResidentSampledNot2d { .. } => "resident_sampled_not_2d",
            Self::GuestRunSampledNot2d { .. } => "guest_run_sampled_not_2d",
            Self::SecondaryAttachmentCap { .. } => "secondary_attachment_cap",
            Self::UnstructuredStateMachineShader { .. } => "unstructured_state_machine_shader",
            Self::InvalidTranslatedModule { .. } => "invalid_translated_module",
            Self::FragmentDescriptorUnbound { .. } => "fragment_descriptor_unbound",
            Self::SamplerAnisotropyUnsupported => "sampler_anisotropy_unsupported",
            Self::SamplerMirrorClampToEdgeUnsupported => "sampler_mirror_clamp_to_edge_unsupported",
            Self::DualSourceBlendUnsupported => "dual_source_blend_unsupported",
            // Deliberately delegates: the translation layer already named the
            // exact format problem, and inventing a second slug here would make
            // the two log lines disagree about one event.
            Self::VertexFormat(reason) => reason.slug(),
            Self::ConstantVertexAttribute => "constant_vertex_attribute",
            Self::InstanceRateDivisorUnsupported { .. } => "instance_rate_divisor_unsupported",
            Self::InstanceRateDivisorOverLimit { .. } => "instance_rate_divisor_over_limit",
            Self::NoCombinedGraphicsComputeQueue => "no_combined_graphics_compute_queue",
            Self::NoHostVisibleMemoryForStaging { .. } => "no_host_visible_memory_for_staging",
            Self::NoHostVisibleMemoryForReadback { .. } => "no_host_visible_memory_for_readback",
            Self::NoHostVisibleMemoryForStats { .. } => "no_host_visible_memory_for_stats",
            Self::NoDeviceLocalMemoryForStorageImage { .. } => {
                "no_device_local_memory_for_storage_image"
            }
            Self::NoDeviceLocalMemoryForSlab { .. } => "no_device_local_memory_for_slab",
            Self::NoDeviceLocalMemoryForMrtSecondary { .. } => {
                "no_device_local_memory_for_mrt_secondary"
            }
            Self::NoDeviceLocalMemoryForDepth { .. } => "no_device_local_memory_for_depth",
            Self::SwapchainUnavailable => "swapchain_unavailable",
            Self::QueueCannotPresent { .. } => "queue_cannot_present",
            Self::SwapchainLacksTransferDst => "swapchain_lacks_transfer_dst",
            Self::SwapchainNoSurfaceFormat => "swapchain_no_surface_format",
            Self::SwapchainNoCompositeAlpha => "swapchain_no_composite_alpha",
        }
    }
}

impl std::fmt::Display for DrawReason {
    /// `reason=<slug>` plus the fields that make the line actionable. A decline
    /// naming only its class leaves the reader without the number that caused
    /// it — which binding, which step rate, which limit.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        match self {
            Self::ResidentSampledNot2d { binding } | Self::GuestRunSampledNot2d { binding } => {
                write!(f, " binding={binding}")
            }
            Self::SecondaryAttachmentCap { requested, cap } => {
                write!(f, " requested={requested} cap={cap}")
            }
            Self::UnstructuredStateMachineShader {
                blocks,
                switch_cases,
            } => write!(f, " blocks={blocks} switch_cases={switch_cases}"),
            Self::FragmentDescriptorUnbound { pipeline_ref }
            | Self::InvalidTranslatedModule { pipeline_ref } => {
                write!(f, " pipe={pipeline_ref}")
            }
            Self::VertexFormat(reason) => write!(f, " value={}", reason.value()),
            Self::InstanceRateDivisorUnsupported { step_rate } => write!(f, " rate={step_rate}"),
            Self::InstanceRateDivisorOverLimit { step_rate, limit } => {
                write!(f, " rate={step_rate} limit={limit}")
            }
            Self::NoHostVisibleMemoryForStaging { memory_type_bits }
            | Self::NoHostVisibleMemoryForReadback { memory_type_bits }
            | Self::NoHostVisibleMemoryForStats { memory_type_bits }
            | Self::NoDeviceLocalMemoryForStorageImage { memory_type_bits }
            | Self::NoDeviceLocalMemoryForSlab { memory_type_bits }
            | Self::NoDeviceLocalMemoryForMrtSecondary { memory_type_bits }
            | Self::NoDeviceLocalMemoryForDepth { memory_type_bits } => {
                write!(f, " memory_type_bits={memory_type_bits:#x}")
            }
            Self::QueueCannotPresent { queue_family } => write!(f, " queue_family={queue_family}"),
            _ => Ok(()),
        }
    }
}

/// Why the CPU-readback path could not materialize a resident target.
///
/// # Why this one is typed even though `Invalid` is still free text
///
/// These are the checks a *caller in `runtime/` classifies on*, and it used to
/// classify by calling `e.to_string().contains(…)` on prose — so the payload
/// wording was load-bearing behaviour that no test covered and no gate could
/// see. Typing them makes the classification a `match` on a variant, which the
/// compiler forces open when a check is added, and makes the emitted
/// `reason=<slug>` the name of the check that actually fired rather than one of
/// four coarse buckets.
///
/// This type used to carry twenty-odd more variants, naming every layout,
/// bounds and ordering check the two guest-page DMA entry points applied before
/// the GPU wrote a frame into guest RAM. Those entry points are gone — the GPU
/// no longer addresses guest pages at all — and so are their checks. What is
/// left is the readback that replaced them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetReadDecline {
    /// The readback's identity is not in the resident registry.
    UnknownIdentity,
    /// The readback's resident has never had content written.
    NoReadyContent,
}

impl Decline for TargetReadDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownIdentity => "read_target_unknown_identity",
            Self::NoReadyContent => "read_target_no_ready_content",
        }
    }
}

impl std::fmt::Display for TargetReadDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[DrawReason] = &[
        DrawReason::ResidentSampledNot2d { binding: 0 },
        DrawReason::GuestRunSampledNot2d { binding: 0 },
        DrawReason::SecondaryAttachmentCap {
            requested: 0,
            cap: 0,
        },
        DrawReason::UnstructuredStateMachineShader {
            blocks: 0,
            switch_cases: 0,
        },
        DrawReason::InvalidTranslatedModule { pipeline_ref: 0 },
        DrawReason::FragmentDescriptorUnbound { pipeline_ref: 0 },
        DrawReason::SamplerAnisotropyUnsupported,
        DrawReason::SamplerMirrorClampToEdgeUnsupported,
        DrawReason::ConstantVertexAttribute,
        DrawReason::InstanceRateDivisorUnsupported { step_rate: 0 },
        DrawReason::InstanceRateDivisorOverLimit {
            step_rate: 0,
            limit: 0,
        },
        DrawReason::NoCombinedGraphicsComputeQueue,
        DrawReason::NoHostVisibleMemoryForStaging {
            memory_type_bits: 0,
        },
        DrawReason::NoHostVisibleMemoryForReadback {
            memory_type_bits: 0,
        },
        DrawReason::NoHostVisibleMemoryForStats {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForStorageImage {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForSlab {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForMrtSecondary {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForDepth {
            memory_type_bits: 0,
        },
        DrawReason::SwapchainUnavailable,
        DrawReason::QueueCannotPresent { queue_family: 0 },
        DrawReason::SwapchainLacksTransferDst,
        DrawReason::SwapchainNoSurfaceFormat,
        DrawReason::SwapchainNoCompositeAlpha,
    ];

    /// The rule this enum exists to enforce: two checks sharing a slug means a
    /// grep of the fail log cannot tell you which one fired.
    #[test]
    fn every_reason_has_its_own_slug() {
        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate DrawReason slug");
    }

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

    /// A decline that names only its class is not actionable — the reader needs
    /// the binding, the rate, the limit.
    #[test]
    fn the_rendered_line_carries_the_load_bearing_fields() {
        assert_eq!(
            DrawReason::ResidentSampledNot2d { binding: 34 }.to_string(),
            "reason=resident_sampled_not_2d binding=34"
        );
        assert_eq!(
            DrawReason::InstanceRateDivisorOverLimit {
                step_rate: 9,
                limit: 4
            }
            .to_string(),
            "reason=instance_rate_divisor_over_limit rate=9 limit=4"
        );
        assert_eq!(
            DrawReason::SecondaryAttachmentCap {
                requested: 9,
                cap: 7
            }
            .to_string(),
            "reason=secondary_attachment_cap requested=9 cap=7"
        );
        // A field-free reason renders just its slug, with no trailing space.
        assert_eq!(
            DrawReason::SwapchainUnavailable.to_string(),
            "reason=swapchain_unavailable"
        );
    }

    /// The memory-type lookups were `DrawError::Vulkan("no host-visible memory
    /// for staging")` — free-text prose rendered as the coarse
    /// `vk_engine_vk_untyped` slug, which classified a *capability* refusal as a
    /// failed Vulkan call. They now name their purpose and carry the requirement
    /// bits that matched no memory type.
    #[test]
    fn a_memory_type_refusal_names_its_purpose_and_carries_the_bits() {
        assert_eq!(
            DrawReason::NoHostVisibleMemoryForStaging {
                memory_type_bits: 0x5
            }
            .to_string(),
            "reason=no_host_visible_memory_for_staging memory_type_bits=0x5"
        );
        assert_eq!(
            DrawReason::NoDeviceLocalMemoryForDepth {
                memory_type_bits: 0x82
            }
            .to_string(),
            "reason=no_device_local_memory_for_depth memory_type_bits=0x82"
        );
        // Staging, readback and stats are three purposes that all want
        // host-visible memory — a shared slug would leave a grep unable to say
        // which allocation had nowhere to live.
        assert_ne!(
            DrawReason::NoHostVisibleMemoryForStaging {
                memory_type_bits: 0
            }
            .slug(),
            DrawReason::NoHostVisibleMemoryForReadback {
                memory_type_bits: 0
            }
            .slug()
        );
        assert_ne!(
            DrawReason::NoHostVisibleMemoryForReadback {
                memory_type_bits: 0
            }
            .slug(),
            DrawReason::NoHostVisibleMemoryForStats {
                memory_type_bits: 0
            }
            .slug()
        );
    }

    #[test]
    fn slab_memory_refusal_carries_its_own_slug_and_bits() {
        let slab = DrawReason::NoDeviceLocalMemoryForSlab {
            memory_type_bits: 0x81,
        };
        assert_eq!(slab.slug(), "no_device_local_memory_for_slab");
        assert_eq!(
            slab.to_string(),
            "reason=no_device_local_memory_for_slab memory_type_bits=0x81"
        );
    }

    /// Two readback checks sharing a slug is the defect this enum replaced —
    /// the prose it grew out of reported distinct faults under one name.
    ///
    /// This case is what is left of a twenty-five-variant sweep that covered
    /// every layout, bounds and ordering check the guest-page DMA entry points
    /// applied. Those checks went out with the DMA; this is not a reduced
    /// version of that gate, it is the gate for the two checks that remain.
    #[test]
    fn every_target_read_decline_has_its_own_slug() {
        const ALL: &[TargetReadDecline] = &[
            TargetReadDecline::UnknownIdentity,
            TargetReadDecline::NoReadyContent,
        ];
        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate TargetReadDecline slug");
        for r in ALL {
            assert_eq!(r.to_string(), format!("reason={}", r.slug()));
            assert!(r.fields().is_empty(), "{r:?} carries no payload");
        }
    }

    /// A vertex-format decline reports the translation layer's own slug rather
    /// than minting a second name for one event.
    #[test]
    fn a_vertex_format_decline_reuses_the_translation_reason() {
        let translate = TranslateReason::FormatNotVertexBuffer(97);
        let reason = DrawReason::VertexFormat(translate);
        assert_eq!(reason.slug(), translate.slug());
        assert_eq!(
            reason.to_string(),
            "reason=format_not_vertex_buffer value=97"
        );
    }
}
