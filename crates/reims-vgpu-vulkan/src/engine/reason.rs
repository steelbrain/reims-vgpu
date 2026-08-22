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

use crate::translate::TranslateReason;
use reims_vgpu_observe::Decline;

/// A request the engine understood and declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReason {
    /// A validator rejected the SPIR-V module this device assembled, so it was
    /// never handed to the driver.
    ///
    /// The validator's prose lives in the `spirv_validate` fail line rather
    /// than in this variant, which is `Copy` and compared by value at every
    /// cache. Unlike its neighbours this is not a capability refusal — it is
    /// this device declining to risk the process on undefined behaviour inside
    /// a driver, and it costs the guest the whole dispatch.
    SpirvInvalid,
    /// A previous process died inside the driver call this request would make,
    /// with these exact modules, so this device will not make it again.
    ///
    /// Distinct from [`Self::SpirvInvalid`] in what it knows: that one is a
    /// module a validator said was malformed, this one is a module a validator
    /// accepted and a driver could not survive. Nothing about the module says
    /// so — the evidence is a breadcrumb the dead process left on disk. See
    /// `driver_breadcrumb::quarantine` for why that is the only admissible
    /// input and how to clear it.
    ///
    /// Fieldless because the variant is `Copy` and compared by value at every
    /// negative cache; the key and the call's description ride the fail line.
    DriverCallQuarantined,
    /// A resident target bound as a sampled image must be a plain 2D image;
    /// arrayed and volume residents have no bind path.
    ResidentSampledNot2d {
        binding: u32,
    },
    /// The guest left a texture slot null and the host cannot bind Vulkan null
    /// descriptors. No ordinary image is substituted because its dimensions
    /// and query behavior would be invented.
    NullSampledImageUnsupported {
        binding: u32,
    },
    /// The guest left a sampler slot null and the host cannot bind a Vulkan
    /// null descriptor. No sampler state is fabricated for the missing object.
    NullSamplerUnsupported {
        binding: u32,
    },
    /// The draw rasterizes into more viewport/scissor slots than the host can
    /// declare in a pipeline.
    ///
    /// `limit` is `maxViewports` where `multiViewport` is advertised and `1`
    /// where it is not, which is why both travel: "the host refused 4" reads
    /// very differently from "the host refused 4 because it offers no multiple
    /// viewports at all", and only the second is a whole missing feature.
    ViewportSlotsUnsupported {
        requested: u32,
        limit: u32,
        multi_viewport: bool,
    },
    /// The draw arms `MTLVisibilityResultModeCounting` and the host cannot
    /// record an exact occlusion count.
    ///
    /// Only the counting arm reaches here. `VK_QUERY_TYPE_OCCLUSION` itself
    /// needs no feature, and an imprecise query is `Boolean` exactly — so
    /// `Boolean` is served on every device and this refusal names the one thing
    /// that is genuinely missing. It carries the feature bit as well as the ask
    /// for the reason `ViewportSlotsUnsupported` above carries
    /// `multi_viewport`: "refused a counting query" and "refused it because
    /// this host offers no precise occlusion at all" are different findings.
    ///
    /// Refusing rather than degrading is the point. Recording without
    /// `PRECISE` would answer a counting guest with a number that is neither
    /// the count nor recognisably wrong.
    VisibilityCountingUnsupported {
        occlusion_query_precise: bool,
    },
    MultisampleAttachmentSampleCountMismatch {
        attachment: u32,
        raster: u32,
    },
    MultisampleResidentTargetMissing {
        sample_count: u32,
    },
    MultisampleLinearTransferUnsupported {
        sample_count: u32,
    },
    MultisampleSampleCountUnsupported {
        requested: u32,
        limit: u32,
    },
    MultisampleStoreActionUnsupported {
        store_action: u16,
    },
    /// A multisample source asks to load prior contents. The current scratch
    /// rail can preserve an already-open encoder, but cannot import a
    /// multisample image from guest linear storage at encoder start.
    MultisampleLoadActionUnsupported {
        load_action: u16,
    },
    MultisampleResolveShapeUnsupported {
        color_targets: u32,
        depth: bool,
        color_input: bool,
    },
    /// Metal clears a depth attachment over its complete image even when a
    /// smaller attachment constrains rasterization, but this Vulkan format
    /// cannot be a transfer destination for the required pre-pass clear.
    AttachmentWideDepthClearUnsupported {
        format: i32,
    },
    /// More MRT secondary attachments than the render pass can carry.
    SecondaryAttachmentCap {
        requested: usize,
        cap: usize,
    },
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
    /// A pixel-coordinate sampler names different minification and
    /// magnification filters. Metal does not define that combination, and the
    /// shader translator cannot choose between them without derivative state.
    SamplerPixelMixedFilters,
    /// A pixel-coordinate sampler asks for mip filtering. Pixel coordinates
    /// address the base level directly, so this is not a state Vulkan can
    /// represent as an unnormalized sampler.
    SamplerPixelMipmapped,
    /// A pixel-coordinate sampler asks for a U/V address mode outside the
    /// clamping modes admitted by the sampler contract.
    SamplerPixelAddressMode,
    /// A pixel-coordinate sampler asks for anisotropic filtering, whose
    /// derivative-dependent result cannot be reconstructed exactly.
    SamplerPixelAnisotropy,
    /// The guest sampler asks for pixel (unnormalized) coordinates **and** a
    /// depth-compare function, and Vulkan forbids the pair outright
    /// (`VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01077`: `compareEnable`
    /// must be `VK_FALSE`).
    ///
    /// The common sampler projection refuses every state it cannot preserve.
    /// This variant names comparison separately because dropping it would
    /// return the sampled value instead of the comparison result.
    SamplerUnnormalizedCompare,
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
    /// The guest asked for `MTLTriangleFillModeLines` and this device does not
    /// advertise `VkPhysicalDeviceFeatures::fillModeNonSolid`, so no pipeline
    /// on it can name `VK_POLYGON_MODE_LINE`.
    ///
    /// The alternative is rasterizing the wireframe filled, which is a whole
    /// pass of wrong pixels the guest is never told about. Same reading as
    /// [`Self::DualSourceBlendUnsupported`]: optional core feature, asked for
    /// at device creation, declined by name where the host says no.
    FillModeNonSolidUnsupported,
    /// Line-rasterized geometry asks for a width other than 1.0 on a device
    /// that does not advertise `wideLines`.
    WideLinesUnsupported {
        requested_bits: u32,
    },
    /// A finite or positive-infinite line width lies outside the inclusive
    /// range reported by the physical device.
    LineWidthOutOfRange {
        requested_bits: u32,
        min_bits: u32,
        max_bits: u32,
    },
    /// A depth-bias component is non-finite and cannot be passed to Vulkan
    /// without undefined rasterization.
    DepthBiasNonFinite {
        component: u8,
        value_bits: u32,
    },
    /// The guest supplied a nonzero clamp but the device did not advertise
    /// `depthBiasClamp`.
    DepthBiasClampUnsupported {
        clamp_bits: u32,
    },
    /// The guest asked for `MTLDepthClipModeClamp` and this device does not
    /// advertise `VkPhysicalDeviceFeatures::depthClamp`, so no pipeline on it
    /// can set `depthClampEnable`.
    ///
    /// Clipping instead discards every fragment the guest asked to keep at the
    /// near and far planes, which is missing geometry rather than shifted
    /// geometry — the sibling of the fill-mode refusal above.
    DepthClampUnsupported,
    /// The primary colour attachment has no faithful Vulkan format on this
    /// backend. Carries the translation reason so the refusal keeps one name.
    ColorAttachmentFormat(TranslateReason),
    /// The device declines this vertex attribute format and no portable
    /// substitute fits. Carries the translation-layer reason so the two log
    /// lines agree on why.
    VertexFormat(TranslateReason),
    /// A constant-rate vertex attribute (`divisor == 0`) on a device without
    /// `vertexAttributeInstanceRateZeroDivisor`.
    ConstantVertexAttribute,
    /// A per-instance step rate above 1 on a device without
    /// `vertexAttributeInstanceRateDivisor`.
    InstanceRateDivisorUnsupported {
        step_rate: u32,
    },
    /// A per-instance step rate above the device's `maxVertexAttribDivisor`.
    InstanceRateDivisorOverLimit {
        step_rate: u32,
        limit: u32,
    },
    /// No queue family supports graphics and compute together, which the
    /// engine's single-queue submit model requires.
    NoCombinedGraphicsComputeQueue,
    /// A shader declares a descriptor array with unpopulated Metal handle
    /// slots, but the host cannot make those slots legally partially bound.
    DescriptorArrayUnsupported {
        binding: u32,
        count: u32,
        unpopulated: u32,
        required_descriptors: u32,
        descriptor_limit: u32,
        partially_bound: bool,
        null_descriptor: bool,
        dynamic_indexing: bool,
    },
    /// Two resources claim one Vulkan binding with incompatible descriptor
    /// type, stage visibility, or array width.
    DescriptorBindingConflict {
        binding: u32,
        first_type: u32,
        first_count: u32,
        second_type: u32,
        second_count: u32,
    },
    // The memory-type lookups. Each is a `memory_type_for(bits, class)` that
    // found nothing: the device advertises no memory type satisfying the buffer
    // or image's requirement bits under the class this allocation needs. That is
    // a device *capability* refusal, not a failed Vulkan call — it matters on the
    // matrix rows nobody here owns, where a class an NVIDIA host offers may be
    // absent. Named per purpose because "which allocation had nowhere to live" is
    // the diagnostic; each carries the requirement bits that matched no type.
    /// No host-visible memory type for a staging (upload) buffer.
    NoHostVisibleMemoryForStaging {
        memory_type_bits: u32,
    },
    /// No host-visible memory type for a readback buffer.
    NoHostVisibleMemoryForReadback {
        memory_type_bits: u32,
    },
    /// No host-visible memory type for the stats-reduction readback buffer.
    NoHostVisibleMemoryForStats {
        memory_type_bits: u32,
    },
    /// No device-local memory type for a storage image.
    NoDeviceLocalMemoryForStorageImage {
        memory_type_bits: u32,
    },
    /// No device-local memory type for a shared optimal-image slab.
    NoDeviceLocalMemoryForSlab {
        memory_type_bits: u32,
    },
    /// No device-local memory type for an MRT secondary attachment image.
    NoDeviceLocalMemoryForMrtSecondary {
        memory_type_bits: u32,
    },
    /// No device-local memory type for a depth attachment image.
    NoDeviceLocalMemoryForDepth {
        memory_type_bits: u32,
    },
    /// No device-local memory type for a draw-time guest gather destination.
    NoDeviceLocalMemoryForGuestGather {
        memory_type_bits: u32,
    },
    /// `VK_KHR_swapchain` is not enabled on the engine device.
    SwapchainUnavailable,
    /// The engine's queue family cannot present to the host window's surface.
    QueueCannotPresent {
        queue_family: u32,
    },
    /// The surface's swapchain images cannot be a transfer destination, which
    /// the present blit requires.
    SwapchainLacksTransferDst,
    /// The surface advertises no formats at all.
    SwapchainNoSurfaceFormat,
    /// The surface advertises no composite-alpha mode.
    SwapchainNoCompositeAlpha,
    /// A binding one of this draw's two modules statically uses is absent from
    /// the descriptor set layout this draw would build.
    ///
    /// The draw-path twin of
    /// [`super::compute_execution::ComputeExecutionDecline::UsedBindingAbsentFromLayout`],
    /// and it is the same host kill: Vulkan requires the pipeline layout to
    /// describe every statically-used resource, and Mesa's Intel driver does not
    /// merely assume that — it scores each used binding as
    /// `(use_count << 7) / array_size` over an array it sized to
    /// `max_binding + 1` and zero-filled, so an absent binding under a declared
    /// one divides by zero and `vkCreateGraphicsPipelines` kills the process
    /// with `SIGFPE` rather than returning an error. There is no status to
    /// inspect afterwards and no guest packet to fail; the process is gone.
    ///
    /// The compute path has carried this backstop since `25051457` and the draw
    /// path did not, which is the `## Before A Broad Sweep` rule about two arms
    /// consuming one wire form: the null-binding pass was ported and the refusal
    /// that catches classes it cannot represent was not.
    ///
    /// **Expected to stay at zero.** `runtime::draw`'s
    /// `frag_unbound_textures_to_bind_null` fills the contract-defined class
    /// before the request is built, and
    /// [`crate::spirv_bind::descriptor_static_use`] answers
    /// `NotDeclared` for anything that is not a `UniformConstant`, so a storage
    /// buffer — whose root that walk cannot resolve — is never refused on a
    /// guess. A firing therefore names a class the null-binding pass does not
    /// cover, and costs one draw rather than the VM.
    UsedBindingAbsentFromLayout {
        binding: u32,
        /// Which of the draw's two modules declared it, so a firing does not
        /// need a second boot to say where to look.
        fragment: bool,
    },
}

impl reims_vgpu_observe::Decline for DrawReason {
    /// Stable slug for `reason=` in the always-on fail log. One per distinct
    /// check, never shared.
    fn slug(&self) -> &'static str {
        match self {
            Self::SpirvInvalid => "spirv_module_invalid",
            Self::UsedBindingAbsentFromLayout { .. } => "draw_used_binding_absent_from_layout",
            Self::DriverCallQuarantined => "driver_call_quarantined",
            Self::ResidentSampledNot2d { .. } => "resident_sampled_not_2d",
            Self::NullSampledImageUnsupported { .. } => "null_sampled_image_unsupported",
            Self::NullSamplerUnsupported { .. } => "null_sampler_unsupported",
            Self::SecondaryAttachmentCap { .. } => "secondary_attachment_cap",
            Self::ViewportSlotsUnsupported { .. } => "viewport_slots_unsupported",
            Self::VisibilityCountingUnsupported { .. } => "visibility_counting_unsupported",
            Self::MultisampleAttachmentSampleCountMismatch { .. } => {
                "multisample_attachment_sample_count_mismatch"
            }
            Self::MultisampleResidentTargetMissing { .. } => "multisample_resident_target_missing",
            Self::MultisampleLinearTransferUnsupported { .. } => {
                "multisample_linear_transfer_unsupported"
            }
            Self::MultisampleSampleCountUnsupported { .. } => {
                "multisample_sample_count_unsupported"
            }
            Self::MultisampleStoreActionUnsupported { .. } => {
                "multisample_store_action_unsupported"
            }
            Self::MultisampleLoadActionUnsupported { .. } => "multisample_load_action_unsupported",
            Self::MultisampleResolveShapeUnsupported { .. } => {
                "multisample_resolve_shape_unsupported"
            }
            Self::AttachmentWideDepthClearUnsupported { .. } => {
                "attachment_wide_depth_clear_unsupported"
            }
            Self::SamplerAnisotropyUnsupported => "sampler_anisotropy_unsupported",
            Self::SamplerMirrorClampToEdgeUnsupported => "sampler_mirror_clamp_to_edge_unsupported",
            Self::SamplerPixelMixedFilters => "sampler_pixel_mixed_filters",
            Self::SamplerPixelMipmapped => "sampler_pixel_mipmapped",
            Self::SamplerPixelAddressMode => "sampler_pixel_address_mode",
            Self::SamplerPixelAnisotropy => "sampler_pixel_anisotropy",
            Self::SamplerUnnormalizedCompare => "sampler_unnormalized_compare",
            Self::DualSourceBlendUnsupported => "dual_source_blend_unsupported",
            Self::FillModeNonSolidUnsupported => "fill_mode_non_solid_unsupported",
            Self::WideLinesUnsupported { .. } => "wide_lines_unsupported",
            Self::LineWidthOutOfRange { .. } => "line_width_out_of_range",
            Self::DepthBiasNonFinite { .. } => "depth_bias_non_finite",
            Self::DepthBiasClampUnsupported { .. } => "depth_bias_clamp_unsupported",
            Self::DepthClampUnsupported => "depth_clamp_unsupported",
            // Deliberately delegates: the translation layer already named the
            // exact format problem, and inventing a second slug here would make
            // the two log lines disagree about one event.
            Self::ColorAttachmentFormat(reason) | Self::VertexFormat(reason) => reason.slug(),
            Self::ConstantVertexAttribute => "constant_vertex_attribute",
            Self::InstanceRateDivisorUnsupported { .. } => "instance_rate_divisor_unsupported",
            Self::InstanceRateDivisorOverLimit { .. } => "instance_rate_divisor_over_limit",
            Self::NoCombinedGraphicsComputeQueue => "no_combined_graphics_compute_queue",
            Self::DescriptorArrayUnsupported { .. } => "descriptor_array_unsupported",
            Self::DescriptorBindingConflict { .. } => "descriptor_binding_conflict",
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
            Self::NoDeviceLocalMemoryForGuestGather { .. } => {
                "no_device_local_memory_for_guest_gather"
            }
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
            Self::UsedBindingAbsentFromLayout { binding, fragment } => {
                let stage = if *fragment { "fragment" } else { "vertex" };
                write!(f, " binding={binding} stage={stage}")
            }
            Self::ResidentSampledNot2d { binding }
            | Self::NullSampledImageUnsupported { binding }
            | Self::NullSamplerUnsupported { binding } => {
                write!(f, " binding={binding}")
            }
            Self::SecondaryAttachmentCap { requested, cap } => {
                write!(f, " requested={requested} cap={cap}")
            }
            Self::ViewportSlotsUnsupported {
                requested,
                limit,
                multi_viewport,
            } => write!(
                f,
                " requested={requested} limit={limit} multi_viewport={}",
                u8::from(*multi_viewport)
            ),
            Self::VisibilityCountingUnsupported {
                occlusion_query_precise,
            } => write!(
                f,
                " occlusion_query_precise={}",
                u8::from(*occlusion_query_precise)
            ),
            Self::MultisampleAttachmentSampleCountMismatch { attachment, raster } => {
                write!(f, " attachment={attachment} raster={raster}")
            }
            Self::MultisampleResidentTargetMissing { sample_count }
            | Self::MultisampleLinearTransferUnsupported { sample_count } => {
                write!(f, " sample_count={sample_count}")
            }
            Self::MultisampleSampleCountUnsupported { requested, limit } => {
                write!(f, " requested={requested} limit={limit}")
            }
            Self::MultisampleStoreActionUnsupported { store_action } => {
                write!(f, " store_action={store_action}")
            }
            Self::MultisampleLoadActionUnsupported { load_action } => {
                write!(f, " load_action={load_action}")
            }
            Self::MultisampleResolveShapeUnsupported {
                color_targets,
                depth,
                color_input,
            } => write!(
                f,
                " color_targets={color_targets} depth={} color_input={}",
                u8::from(*depth),
                u8::from(*color_input)
            ),
            Self::AttachmentWideDepthClearUnsupported { format } => {
                write!(f, " format={format}")
            }
            Self::ColorAttachmentFormat(reason) | Self::VertexFormat(reason) => {
                write!(f, " value={}", reason.value())
            }
            Self::InstanceRateDivisorUnsupported { step_rate } => write!(f, " rate={step_rate}"),
            Self::InstanceRateDivisorOverLimit { step_rate, limit } => {
                write!(f, " rate={step_rate} limit={limit}")
            }
            Self::WideLinesUnsupported { requested_bits } => {
                write!(f, " requested={}", f32::from_bits(*requested_bits))
            }
            Self::LineWidthOutOfRange {
                requested_bits,
                min_bits,
                max_bits,
            } => write!(
                f,
                " requested={} min={} max={}",
                f32::from_bits(*requested_bits),
                f32::from_bits(*min_bits),
                f32::from_bits(*max_bits)
            ),
            Self::DepthBiasNonFinite {
                component,
                value_bits,
            } => write!(
                f,
                " component={component} value={}",
                f32::from_bits(*value_bits)
            ),
            Self::DepthBiasClampUnsupported { clamp_bits } => {
                write!(f, " clamp={}", f32::from_bits(*clamp_bits))
            }
            Self::NoHostVisibleMemoryForStaging { memory_type_bits }
            | Self::NoHostVisibleMemoryForReadback { memory_type_bits }
            | Self::NoHostVisibleMemoryForStats { memory_type_bits }
            | Self::NoDeviceLocalMemoryForStorageImage { memory_type_bits }
            | Self::NoDeviceLocalMemoryForSlab { memory_type_bits }
            | Self::NoDeviceLocalMemoryForMrtSecondary { memory_type_bits }
            | Self::NoDeviceLocalMemoryForDepth { memory_type_bits }
            | Self::NoDeviceLocalMemoryForGuestGather { memory_type_bits } => {
                write!(f, " memory_type_bits={memory_type_bits:#x}")
            }
            Self::QueueCannotPresent { queue_family } => write!(f, " queue_family={queue_family}"),
            Self::DescriptorArrayUnsupported {
                binding,
                count,
                unpopulated,
                required_descriptors,
                descriptor_limit,
                partially_bound,
                null_descriptor,
                dynamic_indexing,
            } => {
                write!(
                    f,
                    " binding={binding} count={count} unpopulated={unpopulated} \
                     required_descriptors={required_descriptors} descriptor_limit={descriptor_limit} \
                     partially_bound={} null_descriptor={} dynamic_indexing={}",
                    u8::from(*partially_bound),
                    u8::from(*null_descriptor),
                    u8::from(*dynamic_indexing)
                )
            }
            Self::DescriptorBindingConflict {
                binding,
                first_type,
                first_count,
                second_type,
                second_count,
            } => write!(
                f,
                " binding={binding} first_type={first_type} first_count={first_count} \
                 second_type={second_type} second_count={second_count}"
            ),
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
    ///
    /// It carries both generations because the bare variant could not be
    /// diagnosed. "Not in the registry" is two findings with opposite repairs
    /// and one word: either **nothing names this surface**, so the guest never
    /// rendered into it or its resident was reclaimed under the caller, or the
    /// **target is there under a different generation**, so the key this caller
    /// built and the key the draw registered disagree and the resident is
    /// sitting untouched beside the lookup that missed it.
    ///
    /// `held` is `ResourcePools::registry_generation_near`: the generation of a
    /// registry entry matching this identity in everything else, or `None` when
    /// there is none. A `Some` that differs from `asked` is the second finding,
    /// stated rather than inferred — which is what the bare variant made
    /// impossible when the copied-resident route lost every Maps frame to it.
    ///
    /// `prior` closes the first finding the same way. `how == Absent` says the
    /// registry names this surface under no key at all, which is still two
    /// findings — the guest never rendered into it, or **this device took its
    /// resident away** — and `ResourcePools::prior_reclaim` has held the answer
    /// to the second the whole time. The sampled rail already quotes it as
    /// `prior=`; the readback did not, so 132 latched refusals on one driven
    /// import-off macos-13 boot said a resident was gone and none of them said
    /// who removed it. `None` is the honest "no record", which covers both
    /// "never held one" and "reclaimed longer ago than the history window
    /// reaches" — that method deliberately does not guess between them.
    UnknownIdentity {
        asked: u64,
        held: Option<u64>,
        how: super::types::TargetKeyDivergence,
        prior: Option<super::types::ResidentReclaim>,
    },
    /// The readback's resident has never had content written.
    NoReadyContent,
    /// Vulkan cannot copy a multisample image directly to a buffer. The image
    /// remains usable by GPU consumers; a host serialization needs an explicit
    /// shader resolve chosen by the caller's sample semantics.
    MultisampleImage { sample_count: u32 },
    /// The readback's resident does not hold four-byte texels.
    ///
    /// Every consumer of a [`super::TargetReadback`] speaks RGBA8 —
    /// `into_rgba8` exchanges channels in `chunks_exact_mut(4)`, and the CPU
    /// Store rail converts from RGBA8 a row at a time — and the readback buffer
    /// is sized `w * h * 4` for the same reason. A wider resident delivered
    /// through here would be read as the wrong texel with nothing to say so.
    ///
    /// A refusal rather than a conversion, because these rails are the
    /// *fallback* for a target whose frame the GPU could not write into guest
    /// pages directly. A resident wide enough to trip this is one the direct
    /// rail handles, so the honest answer is to name the gap rather than to
    /// narrow the frame on the way through it.
    TexelNotFourBytes { format: ash::vk::Format },
}

impl Decline for TargetReadDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownIdentity { .. } => "read_target_unknown_identity",
            Self::NoReadyContent => "read_target_no_ready_content",
            Self::MultisampleImage { .. } => "read_target_multisample_image",
            Self::TexelNotFourBytes { .. } => "read_target_texel_not_four_bytes",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownIdentity {
                asked,
                held,
                how,
                prior,
            } => vec![
                ("diverges", how.label().to_string()),
                ("asked_gen", asked.to_string()),
                (
                    "held_gen",
                    held.map_or_else(|| "none".to_string(), |g| g.to_string()),
                ),
                ("prior", prior.map_or("none", |why| why.slug()).to_string()),
            ],
            Self::NoReadyContent => Vec::new(),
            Self::MultisampleImage { sample_count } => {
                vec![("sample_count", sample_count.to_string())]
            }
            Self::TexelNotFourBytes { format } => vec![("format", format!("{format:?}"))],
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
        DrawReason::SecondaryAttachmentCap {
            requested: 0,
            cap: 0,
        },
        DrawReason::ViewportSlotsUnsupported {
            requested: 0,
            limit: 0,
            multi_viewport: false,
        },
        DrawReason::VisibilityCountingUnsupported {
            occlusion_query_precise: false,
        },
        DrawReason::MultisampleAttachmentSampleCountMismatch {
            attachment: 4,
            raster: 2,
        },
        DrawReason::MultisampleResidentTargetMissing { sample_count: 4 },
        DrawReason::MultisampleLinearTransferUnsupported { sample_count: 4 },
        DrawReason::MultisampleSampleCountUnsupported {
            requested: 4,
            limit: 1,
        },
        DrawReason::MultisampleStoreActionUnsupported { store_action: 3 },
        DrawReason::MultisampleLoadActionUnsupported { load_action: 1 },
        DrawReason::MultisampleResolveShapeUnsupported {
            color_targets: 2,
            depth: false,
            color_input: false,
        },
        DrawReason::SamplerAnisotropyUnsupported,
        DrawReason::SamplerMirrorClampToEdgeUnsupported,
        DrawReason::SamplerPixelMixedFilters,
        DrawReason::SamplerPixelMipmapped,
        DrawReason::SamplerPixelAddressMode,
        DrawReason::SamplerPixelAnisotropy,
        DrawReason::SamplerUnnormalizedCompare,
        DrawReason::ConstantVertexAttribute,
        DrawReason::InstanceRateDivisorUnsupported { step_rate: 0 },
        DrawReason::InstanceRateDivisorOverLimit {
            step_rate: 0,
            limit: 0,
        },
        DrawReason::NoCombinedGraphicsComputeQueue,
        DrawReason::DescriptorArrayUnsupported {
            binding: 0,
            count: 2,
            unpopulated: 1,
            required_descriptors: 2,
            descriptor_limit: 1,
            partially_bound: false,
            null_descriptor: false,
            dynamic_indexing: false,
        },
        DrawReason::DescriptorBindingConflict {
            binding: 0,
            first_type: 0,
            first_count: 1,
            second_type: 1,
            second_count: 2,
        },
        DrawReason::AttachmentWideDepthClearUnsupported { format: 126 },
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
        DrawReason::DualSourceBlendUnsupported,
        DrawReason::FillModeNonSolidUnsupported,
        DrawReason::WideLinesUnsupported {
            requested_bits: 2.0f32.to_bits(),
        },
        DrawReason::LineWidthOutOfRange {
            requested_bits: 65.0f32.to_bits(),
            min_bits: 1.0f32.to_bits(),
            max_bits: 64.0f32.to_bits(),
        },
        DrawReason::DepthBiasNonFinite {
            component: 0,
            value_bits: f32::NAN.to_bits(),
        },
        DrawReason::DepthBiasClampUnsupported {
            clamp_bits: 1.0f32.to_bits(),
        },
        DrawReason::DepthClampUnsupported,
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
            TargetReadDecline::UnknownIdentity {
                asked: 7,
                held: None,
                how: crate::engine::TargetKeyDivergence::Absent,
                prior: None,
            },
            TargetReadDecline::NoReadyContent,
            TargetReadDecline::MultisampleImage { sample_count: 2 },
        ];
        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate TargetReadDecline slug");
        for r in ALL {
            assert_eq!(r.to_string(), format!("reason={}", r.slug()));
            if matches!(r, TargetReadDecline::NoReadyContent) {
                assert!(r.fields().is_empty(), "{r:?} carries no payload");
            }
        }
    }

    /// The two findings an absent registry entry can carry are distinguishable
    /// on the line. They were one word until a driven boot lost every Maps
    /// frame to this refusal and nothing in the log could say which of them it
    /// was.
    ///
    /// `prior` is the third: `diverges=absent` is itself two findings, and this
    /// is the one where **this device** took the resident away rather than the
    /// guest never having created it. A `none` there is a real "no record" and
    /// not a claim that nothing was reclaimed.
    #[test]
    fn an_absent_resident_says_whether_the_target_exists_under_another_key() {
        use crate::engine::TargetKeyDivergence;
        let nothing = TargetReadDecline::UnknownIdentity {
            asked: 4,
            held: None,
            how: TargetKeyDivergence::Absent,
            prior: Some(crate::engine::types::ResidentReclaim::ResourceReleased),
        };
        let stale = TargetReadDecline::UnknownIdentity {
            asked: 4,
            held: Some(5),
            how: TargetKeyDivergence::Generation,
            prior: None,
        };
        // One slug, because it is one check. What separates them is the
        // payload, which is what `Emit::decline` puts on the line — `Display`
        // renders the slug alone and always has, so asserting on it here would
        // pass whatever the fields said.
        assert_eq!(nothing.slug(), stale.slug());
        assert_eq!(
            nothing.fields(),
            vec![
                ("diverges", "absent".to_string()),
                ("asked_gen", "4".to_string()),
                ("held_gen", "none".to_string()),
                ("prior", "resource_released".to_string())
            ]
        );
        assert_eq!(
            stale.fields(),
            vec![
                ("diverges", "generation".to_string()),
                ("asked_gen", "4".to_string()),
                ("held_gen", "5".to_string()),
                ("prior", "none".to_string())
            ]
        );
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
