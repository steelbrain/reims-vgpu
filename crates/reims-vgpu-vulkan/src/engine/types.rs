//! Draw request surface for the internal Vulkan engine (v1 §1.2 surface).
//!
//! Field meanings match the historical Metal→Vulkan product draw seam
//! (blend, Load seed, stage-in attributes, SSBOs, sampled images).

#[cfg(test)]
use crate::translate;
pub use reims_vgpu_core::{
    viewport_slot_count, AttachmentInitial, AttachmentSlot, BlendFactor, BlendOp,
    BlendStateResource, BufferContent, ColorLoadAction, ComputeBufferBacking, ComputeBufferOutput,
    ComputeBufferResource, ComputeBufferResult, ComputeImageDestination, ComputeImageResult,
    ComputeOutput, ComputeRequest, ComputeResidentSampleBind, ComputeSampledImageResource,
    ComputeSampledImageSource, ComputeStorageImageResource, ComputeStorageImageSeed,
    ComputeStorageResidency, CullMode, DepthAttachment, DepthClipMode, DepthState, DrawOutput,
    DrawRequest, FillMode, IndexType, IndexedDrawResource, PrimitiveTopology, SampledByteOrigin,
    SampledContentIdentity, SampledImageResource, SampledSource, SamplerAddressMode,
    SamplerBorderColor, SamplerCompareFunction, SamplerFilter, SamplerMipFilter, SamplerResource,
    ScissorResource, SecondaryColorTarget, SeedOrder, StencilAttachment, StencilFaceOps, StencilOp,
    StencilState, StorageBufferResource, VertexAttributeFormat, VertexAttributeResource,
    VertexStepFunction, ViewportResource, VisibilityResultMode,
};
pub use reims_vgpu_memory::{
    GuestImageSource, GuestRun, GuestRunSource, GuestTargetBacking, GuestTargetMemory,
    GuestTargetPlan, GuestTargetSeed, WindowStretch,
};
pub use reims_vgpu_protocol::ColorWriteMask;
pub use reims_vgpu_protocol::StorageImageFormat;

/// Named engine failure. Stable prefixes for observe greps (`vk_engine_*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawError {
    /// Init / ICD / device selection failed. Latched by `ContextOwner`, except
    /// when it is out of memory — see `ContextOwner::note_init_failure`.
    Init(super::init_decline::InitDecline),
    /// Understood but declined — a capability this device or this engine does
    /// not have. Typed so each distinct check carries its own `reason=` slug;
    /// see [`super::reason::DrawReason`].
    Unsupported(super::reason::DrawReason),
    /// Engine façade or host-window presenter state changed under a valid
    /// request, or a façade input cannot describe a scanout.
    Facade(super::facade_decline::EngineFacadeDecline),
    /// Runtime pipeline/MTLB/AIR preparation failed before an engine request
    /// could be validated.
    DrawPreparation(super::draw_preparation::DrawPreparationDecline),
    /// Draw request rejected before context creation or GPU work.
    DrawValidation(super::draw_validation::DrawValidationDecline),
    /// A validated draw request failed while materializing execution state.
    DrawExecution(super::draw_execution::DrawExecutionDecline),
    /// Compute request rejected before context creation or GPU work.
    ComputeValidation(super::compute_validation::ComputeValidationDecline),
    /// A validated compute request lost or mismatched resident execution state.
    ComputeExecution(super::compute_execution::ComputeExecutionDecline),
    /// A resident-target readback could not find its content.
    /// See [`super::reason::TargetReadDecline`].
    TargetRead(super::reason::TargetReadDecline),
    /// A resident's frame could not be copied straight into the guest's pages,
    /// so the flush owes the CPU route instead.
    /// See [`super::host_ram::GuestWriteDecline`].
    GuestPageWrite(super::host_ram::GuestWriteDecline),
    /// A specific Vulkan call that returned an error, typed by *(rail,
    /// operation)*. Former `Vulkan(String)` sites move here so the log names
    /// which call refused.
    /// See [`super::vk_call::VkCall`].
    VkCall(super::vk_call::VkCall),
    /// The image-memory slab rejected an impossible allocation/invariant
    /// without pretending the driver returned OOM.
    Slab(super::slab::SlabDecline),
    /// Fence wait timed out.
    FenceTimeout,
    /// The session-wide native-object retirement sequence exhausted its exact
    /// identity space before a command buffer began recording.
    RecordingSequenceExhausted,
    /// Device lost and recreate budget exhausted (or mid-draw loss).
    DeviceLost(super::device_lost::DeviceLostDecline),
}

impl DrawError {
    /// Whether this refusal is the device saying it has no memory left, as
    /// opposed to refusing for any other reason.
    ///
    /// The one class worth retrying: it is a statement about how much memory is
    /// in use at this instant rather than about the request, so giving memory
    /// back can change the answer. Every other `DrawError` describes something
    /// about the request or the driver that a second identical attempt would
    /// meet again.
    ///
    /// Both Vulkan out-of-memory results count. `ERROR_OUT_OF_HOST_MEMORY` is
    /// included because this device's pools hold host allocations too — the
    /// HOST_VISIBLE staging and readback rings — so the same reclaim is the
    /// right response to either. `ERROR_DEVICE_LOST` deliberately is not: it has
    /// its own variant and is answered by recreating the context, and retrying
    /// an allocation against a lost device would only fail again.
    ///
    /// [`Self::Init`] answers here too, and it is the arm with the widest blast
    /// radius. `vkCreateInstance` and `vkCreateDevice` both refuse with
    /// `ERROR_OUT_OF_HOST_MEMORY`, and bring-up is latched by
    /// `ContextOwner::init_error` — so a host that was momentarily short of RAM
    /// at the first draw would otherwise take the whole Vulkan engine down for
    /// the life of the process. The bring-up checks this device decides itself
    /// (no loader, no device, no graphics queue, below the API floor) carry no
    /// result and are correctly permanent.
    pub fn out_of_memory(&self) -> bool {
        let result = match self {
            Self::VkCall(c) => Some(c.result),
            Self::Init(d) => d.vk_result(),
            _ => None,
        };
        matches!(
            result,
            Some(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                | Some(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        )
    }
}

impl std::fmt::Display for DrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(d) => write!(f, "vk_engine_init: {d}"),
            Self::Unsupported(r) => write!(f, "vk_engine_unsupported: {r}"),
            Self::Facade(d) => write!(f, "vk_engine_facade: {d}"),
            Self::DrawPreparation(d) => write!(f, "vk_engine_draw_preparation: {d}"),
            Self::DrawValidation(d) => write!(f, "vk_engine_draw_validation: {d}"),
            Self::DrawExecution(d) => write!(f, "vk_engine_draw_execution: {d}"),
            Self::ComputeValidation(d) => write!(f, "vk_engine_compute_validation: {d}"),
            Self::ComputeExecution(d) => write!(f, "vk_engine_compute_execution: {d}"),
            Self::TargetRead(d) => write!(f, "vk_engine_target_read: {d}"),
            Self::GuestPageWrite(d) => write!(f, "vk_engine_guest_page_write: {d}"),
            Self::VkCall(c) => write!(f, "vk_engine_vk: {c}"),
            Self::Slab(d) => write!(f, "vk_engine_slab: {d}"),
            Self::FenceTimeout => write!(f, "vk_engine_fence_timeout"),
            Self::RecordingSequenceExhausted => {
                write!(f, "vk_engine_recording_sequence_exhausted")
            }
            Self::DeviceLost(d) => write!(f, "vk_engine_device_lost: {d}"),
        }
    }
}

impl std::error::Error for DrawError {}

impl reims_vgpu_observe::Decline for DrawError {
    /// Every variant delegates to the typed decline that names its check, so
    /// one event has one reason at every layer.
    fn slug(&self) -> &'static str {
        match self {
            Self::TargetRead(d) => d.slug(),
            Self::GuestPageWrite(d) => d.slug(),
            Self::Unsupported(r) => r.slug(),
            // Delegates like the two typed variants above: the call names itself,
            // so one event has one name whether it is read here or on `VkCall`.
            Self::VkCall(c) => c.slug(),
            Self::Slab(d) => d.slug(),
            Self::FenceTimeout => "vk_engine_fence_timeout",
            Self::RecordingSequenceExhausted => "vk_engine_recording_sequence_exhausted",
            Self::Init(d) => d.slug(),
            Self::Facade(d) => d.slug(),
            Self::DrawPreparation(d) => d.slug(),
            Self::DrawValidation(d) => d.slug(),
            Self::DrawExecution(d) => d.slug(),
            Self::ComputeValidation(d) => d.slug(),
            Self::ComputeExecution(d) => d.slug(),
            Self::DeviceLost(d) => d.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::TargetRead(d) => d.fields(),
            Self::GuestPageWrite(d) => d.fields(),
            Self::Unsupported(r) => r.fields(),
            Self::VkCall(c) => c.fields(),
            Self::Slab(d) => d.fields(),
            Self::Init(d) => d.fields(),
            Self::DrawValidation(d) => d.fields(),
            Self::DrawExecution(d) => d.fields(),
            Self::ComputeValidation(d) => d.fields(),
            Self::ComputeExecution(d) => d.fields(),
            Self::DeviceLost(d) => d.fields(),
            Self::FenceTimeout => Vec::new(),
            Self::RecordingSequenceExhausted => Vec::new(),
            Self::Facade(d) => d.fields(),
            Self::DrawPreparation(d) => d.fields(),
        }
    }
}

impl From<DrawError> for String {
    fn from(e: DrawError) -> Self {
        e.to_string()
    }
}

/// Descriptor binding of the attachment-0 framebuffer-fetch input attachment.
///
/// Attachment-0 framebuffer-fetch binding in metal2vulkan's selected render
/// descriptor layout. Color inputs are fragment-only, so the stage-separation
/// layout deliberately retains the translator's default range.
pub const COLOR_INPUT_BINDING: u32 = metal2vulkan::reflect::COLOR_INPUT_BINDING_BASE;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct SamplerStateKey {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    pub compare_function: SamplerCompareFunction,
    pub lod_min: u32,
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

pub(crate) fn sampler_state_key(sampler: &SamplerResource) -> SamplerStateKey {
    SamplerStateKey {
        min_filter: sampler.min_filter,
        mag_filter: sampler.mag_filter,
        mip_filter: sampler.mip_filter,
        address_mode_u: sampler.address_mode_u,
        address_mode_v: sampler.address_mode_v,
        address_mode_w: sampler.address_mode_w,
        border_color: sampler.border_color,
        compare_function: sampler.compare_function,
        lod_min: sampler.lod_min,
        lod_max: sampler.lod_max,
        max_anisotropy: sampler.max_anisotropy,
        unnormalized_coordinates: sampler.unnormalized_coordinates,
    }
}

/// Resolve the sampler state whose semantics both shader translation and the
/// Vulkan descriptor must implement.
///
/// Pixel-coordinate sampling has no mip selection, so its LOD clamps do not
/// participate in the result even though they remain part of the serialized
/// sampler descriptor. Vulkan requires both clamps to be zero for an
/// unnormalized sampler. The other Vulkan restrictions are also restrictions
/// of the pixel-sampler contract; refusing violations preserves the requested
/// state instead of silently replacing filters, addressing, or anisotropy.
pub(crate) fn effective_sampler_state(
    sampler: &SamplerResource,
) -> Result<SamplerStateKey, super::reason::DrawReason> {
    effective_sampler_state_key(sampler_state_key(sampler))
}

pub(crate) fn effective_sampler_state_key(
    mut key: SamplerStateKey,
) -> Result<SamplerStateKey, super::reason::DrawReason> {
    use super::reason::DrawReason;
    if !key.unnormalized_coordinates {
        return Ok(key);
    }
    if key.min_filter != key.mag_filter {
        return Err(DrawReason::SamplerPixelMixedFilters);
    }
    if key.mip_filter != SamplerMipFilter::NotMipmapped {
        return Err(DrawReason::SamplerPixelMipmapped);
    }
    if !matches!(
        key.address_mode_u,
        SamplerAddressMode::ClampToEdge
            | SamplerAddressMode::ClampToZero
            | SamplerAddressMode::ClampToBorderColor
    ) || !matches!(
        key.address_mode_v,
        SamplerAddressMode::ClampToEdge
            | SamplerAddressMode::ClampToZero
            | SamplerAddressMode::ClampToBorderColor
    ) {
        return Err(DrawReason::SamplerPixelAddressMode);
    }
    if key.max_anisotropy != 1 {
        return Err(DrawReason::SamplerPixelAnisotropy);
    }
    if key.compare_function != SamplerCompareFunction::Never {
        return Err(DrawReason::SamplerUnnormalizedCompare);
    }
    key.lod_min = 0.0f32.to_bits();
    key.lod_max = 0.0f32.to_bits();
    Ok(key)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BlendKey {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
}

impl BlendKey {
    pub(crate) fn uses_constants(self) -> bool {
        [
            self.src_color,
            self.dst_color,
            self.src_alpha,
            self.dst_alpha,
        ]
        .into_iter()
        .any(BlendFactor::uses_blend_constant)
    }
}

pub(crate) fn blend_key(blend: &BlendStateResource) -> BlendKey {
    BlendKey {
        src_color: blend.src_color,
        dst_color: blend.dst_color,
        color_op: blend.color_op,
        src_alpha: blend.src_alpha,
        dst_alpha: blend.dst_alpha,
        alpha_op: blend.alpha_op,
    }
}

// ---------------------------------------------------------------------------
// Compute request surface
// ---------------------------------------------------------------------------

/// Named compute failure. Same `vk_engine_*` prefix family as draw.
pub type ComputeError = DrawError;

// ---------------------------------------------------------------------------
// Draw residency (workstream D)
// ---------------------------------------------------------------------------

pub use reims_vgpu_core::{TargetIdentity, TargetKeyDivergence};

pub use reims_vgpu_core::ResidentReclaim;

pub type PresentRect = (u32, u32, u32, u32);

pub use reims_vgpu_core::PreparedPresentation as WindowPresentSource;

#[cfg(test)]
mod tests {
    use super::*;

    /// A draw that samples one of its own attachments must reach the snapshot
    /// arm, and "its own" is every attachment it binds rather than slot 0.
    ///
    /// The secondary and depth cases below are the ones that fail against the
    /// primary-only test this replaced, which is what makes them worth writing:
    /// each was a live attachment feedback loop handed to the driver.
    #[test]
    fn a_draw_samples_its_own_attachment_on_every_slot_that_can_carry_one() {
        let surface = |id: u32| TargetIdentity::Surface {
            id,
            width: 64,
            height: 64,
            generation: 0,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };

        let mut req = DrawRequest {
            target_identity: Some(surface(1)),
            ..DrawRequest::default()
        };
        assert!(req.writes_attachment(&surface(1)), "primary colour");
        assert!(
            !req.writes_attachment(&surface(9)),
            "a target this draw does not bind is not a feedback loop, and \
             routing it through the snapshot would cost a copy per draw"
        );

        req.secondary_targets.push(SecondaryColorTarget {
            identity: surface(2),
            target_guest: None,
            width: 64,
            height: 64,
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Bgra8,
            ),
            clear: [0.0; 4],
            load_action: ColorLoadAction::Clear,
            blend: None,
            color_write_mask: ColorWriteMask::default(),
        });
        assert!(req.writes_attachment(&surface(2)), "MRT secondary");
        assert_eq!(
            req.attachment_slot(&surface(2)),
            Some(AttachmentSlot::Secondary),
            "the census has to be able to say which slot matched"
        );
        assert_eq!(req.color_attachment_index(&surface(1)), Some(0));
        assert_eq!(req.color_attachment_index(&surface(2)), Some(1));
        assert_eq!(
            req.attachment_slot(&surface(1)),
            Some(AttachmentSlot::Primary)
        );

        req.depth_attachment = Some(DepthAttachment {
            identity: surface(3),
            resource_lifetime: reims_vgpu_core::ResourceLifetime::new().reference(),
            depth: Some(reims_vgpu_core::DepthAspectAttachment {
                load_action: reims_vgpu_protocol::pass_action::LoadAction::Clear,
                store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
                clear_value: 1.0,
            }),
            stencil: None,
        });
        assert!(req.writes_attachment(&surface(3)), "depth");
        assert_eq!(
            req.attachment_slot(&surface(3)),
            Some(AttachmentSlot::Depth)
        );
        assert_eq!(req.attachment_slot(&surface(9)), None);
        // Three distinct routes, so a census reading one of them cannot be a
        // different slot's population.
        let routes = [
            AttachmentSlot::Primary,
            AttachmentSlot::Secondary,
            AttachmentSlot::Depth,
        ]
        .map(AttachmentSlot::sampled_self_route);
        assert_eq!(
            routes
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );

        // The generation is part of the identity, so a resident the guest has
        // since rewritten is a different target and not this draw's attachment.
        assert!(!req.writes_attachment(&TargetIdentity::Surface {
            id: 1,
            width: 64,
            height: 64,
            generation: 1,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }));
    }

    #[test]
    fn indexed_draw_widths_are_the_fixed_function_element_widths() {
        assert_eq!(IndexType::U16.byte_size(), 2);
        assert_eq!(IndexType::U32.byte_size(), 4);
    }

    #[test]
    fn sampler_cache_state_excludes_binding_but_preserves_sampler_state() {
        let first = SamplerResource::normalized_default(3);
        let mut rebound = SamplerResource::normalized_default(27);
        assert_eq!(sampler_state_key(&first), sampler_state_key(&rebound));
        assert_eq!(first.lod_min_f32(), 0.0);
        assert_eq!(first.lod_max_f32(), f32::MAX);

        rebound.address_mode_v = SamplerAddressMode::Repeat;
        assert_ne!(sampler_state_key(&first), sampler_state_key(&rebound));
    }

    fn valid_pixel_sampler() -> SamplerResource {
        let mut sampler = SamplerResource::normalized_default(0);
        sampler.unnormalized_coordinates = true;
        sampler.lod_min = 1.0f32.to_bits();
        sampler.lod_max = 8.0f32.to_bits();
        sampler
    }

    #[test]
    fn pixel_sampler_projection_changes_only_semantically_inactive_lod_clamps() {
        let sampler = valid_pixel_sampler();
        let raw = sampler_state_key(&sampler);
        let effective = effective_sampler_state(&sampler).expect("valid pixel sampler");
        assert_eq!(f32::from_bits(effective.lod_min), 0.0);
        assert_eq!(f32::from_bits(effective.lod_max), 0.0);
        assert_eq!(
            SamplerStateKey {
                lod_min: raw.lod_min,
                lod_max: raw.lod_max,
                ..effective
            },
            raw
        );
    }

    #[test]
    fn pixel_sampler_projection_refuses_state_it_cannot_preserve() {
        use reims_vgpu_observe::Decline as _;

        let mut sampler = valid_pixel_sampler();
        sampler.min_filter = SamplerFilter::Nearest;
        assert_eq!(
            effective_sampler_state(&sampler).unwrap_err().slug(),
            "sampler_pixel_mixed_filters"
        );

        let mut sampler = valid_pixel_sampler();
        sampler.mip_filter = SamplerMipFilter::Linear;
        assert_eq!(
            effective_sampler_state(&sampler).unwrap_err().slug(),
            "sampler_pixel_mipmapped"
        );

        let mut sampler = valid_pixel_sampler();
        sampler.address_mode_u = SamplerAddressMode::Repeat;
        assert_eq!(
            effective_sampler_state(&sampler).unwrap_err().slug(),
            "sampler_pixel_address_mode"
        );

        let mut sampler = valid_pixel_sampler();
        sampler.max_anisotropy = 2;
        assert_eq!(
            effective_sampler_state(&sampler).unwrap_err().slug(),
            "sampler_pixel_anisotropy"
        );
    }

    #[test]
    fn target_identity_accessors_never_infer_anonymous_geometry() {
        let surface = TargetIdentity::Surface {
            id: 7,
            width: 1920,
            height: 1080,
            generation: 4,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        assert_eq!(
            (surface.width(), surface.height(), surface.generation()),
            (1920, 1080, 4)
        );
        let anonymous = TargetIdentity::Anonymous { slot: 99 };
        assert_eq!(
            (
                anonymous.width(),
                anonymous.height(),
                anonymous.generation()
            ),
            (0, 0, 0)
        );
        assert_eq!(
            TargetIdentity::default(),
            TargetIdentity::Anonymous { slot: 0 }
        );
    }

    /// Re-generation changes the generation and nothing else, on every variant
    /// that has one — which is what lets "is this the same target under a newer
    /// key?" be asked with `PartialEq` instead of a field-by-field comparison
    /// that a new field would silently fall out of.
    #[test]
    fn re_generation_moves_only_the_generation() {
        let all = [
            TargetIdentity::Surface {
                id: 7,
                width: 1920,
                height: 1080,
                generation: 4,
                format: reims_vgpu_protocol::TexelLayout::Bgra8,
            },
            TargetIdentity::Texture {
                ref_: 12,
                width: 64,
                height: 64,
                generation: 4,
                stencil: true,
            },
            TargetIdentity::Gva {
                gva: 0xdead_0000,
                width: 8,
                height: 8,
                generation: 4,
                format: reims_vgpu_protocol::TexelLayout::Bgra8,
            },
        ];
        for identity in &all {
            let moved = identity.with_generation(9);
            assert_eq!(moved.generation(), 9, "{identity:?}");
            assert_ne!(&moved, identity, "{identity:?}");
            // The round trip is the whole claim: everything but the generation
            // survived, so equality after restoring it is field-complete.
            assert_eq!(&moved.with_generation(identity.generation()), identity);
        }
        // `Anonymous` carries no generation, so it is returned as itself rather
        // than being given one it has nowhere to keep.
        let anonymous = TargetIdentity::Anonymous { slot: 99 };
        assert_eq!(anonymous.with_generation(9), anonymous);
    }

    /// The four ways a registry key can miss are told apart, and the ladder is
    /// answered coarsest-first: two identities in different namespaces are not
    /// about one object, so nothing finer about them is reported. A miss that
    /// named none of these sent one session hunting the generation case, which
    /// turned out to be the minority.
    #[test]
    fn a_registry_miss_names_which_field_moved() {
        let asked = TargetIdentity::Surface {
            id: 7,
            width: 1920,
            height: 1080,
            generation: 2,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        };
        assert_eq!(
            asked.diverges_from(&asked.with_generation(1)),
            TargetKeyDivergence::Generation
        );
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Surface {
                id: 7,
                width: 1920,
                height: 900,
                generation: 2,
                format: reims_vgpu_protocol::TexelLayout::Bgra8,
            }),
            TargetKeyDivergence::Geometry
        );
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Texture {
                ref_: 7,
                width: 1920,
                height: 1080,
                generation: 2,
                stencil: false,
            }),
            TargetKeyDivergence::Namespace
        );
        // A format change is what is left once the object, the extent and the
        // generation all agree — and so is any field this enum gains, which is
        // the point of asking the last question with `PartialEq`.
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Surface {
                id: 7,
                width: 1920,
                height: 1080,
                generation: 2,
                format: reims_vgpu_protocol::TexelLayout::Rgba16Float,
            }),
            TargetKeyDivergence::Other
        );
        // Namespace outranks everything: a texture ref that happens to equal a
        // mapping id must not be reported as a resize of it.
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Texture {
                ref_: 7,
                width: 8,
                height: 8,
                generation: 99,
                stencil: false,
            }),
            TargetKeyDivergence::Namespace
        );
    }

    #[test]
    fn storage_format_texel_sizes_cover_every_format_variant() {
        let cases = [
            (StorageImageFormat::Rgba32Float, 16),
            (StorageImageFormat::Rgba16Float, 8),
            (StorageImageFormat::R16Float, 2),
            (StorageImageFormat::Rgba16Uint, 8),
            (StorageImageFormat::Rgba8Uint, 4),
            (StorageImageFormat::Rgba8Sint, 4),
            (StorageImageFormat::Rgba8Unorm, 4),
            (StorageImageFormat::Bgra8Unorm, 4),
            (StorageImageFormat::Rg16Float, 4),
            (StorageImageFormat::R8Unorm, 1),
            (StorageImageFormat::Rg8Unorm, 2),
            (StorageImageFormat::Rgba32Uint, 16),
            (StorageImageFormat::R32Uint, 4),
            (StorageImageFormat::R32Sint, 4),
            (StorageImageFormat::R32Float, 4),
            (StorageImageFormat::Rgb9e5Ufloat, 4),
        ];
        for (format, expected) in cases {
            assert_eq!(format.bytes_per_texel(), expected);
        }
    }

    #[test]
    fn byte_buffer_content_reports_and_borrows_its_exact_payload() {
        let content = BufferContent::from(vec![1, 2, 3, 4]);
        assert_eq!(content.len(), 4);
        assert!(!content.is_empty());
        assert_eq!(content.cpu_bytes().as_ref(), &[1, 2, 3, 4]);
        assert!(BufferContent::from(Vec::new()).is_empty());
    }

    #[test]
    fn default_requests_keep_optional_product_paths_disabled() {
        let draw = DrawRequest::default();
        assert_eq!((draw.width, draw.height, draw.vertex_count), (0, 0, 0));
        assert_eq!(draw.primitive_topology, PrimitiveTopology::Triangle);
        assert_eq!(draw.cull_mode, CullMode::None);
        assert!(draw.target_identity.is_none());
        assert!(draw.depth.is_none());
        assert!(!draw.skip_readback);
        assert!(!draw.color_input);

        let compute = ComputeRequest::default();
        assert_eq!(compute.dispatch.counts, [0, 0, 0]);
        assert_eq!(compute.dispatch.threads_per_grid, [0, 0, 0]);
        assert!(compute.storage_buffers.is_empty());
        assert!(compute.storage_images.is_empty());
    }

    /// The order is a property of the identity, and the three answers matter for
    /// different reasons.
    ///
    /// `Surface` answers from the format its mapping declared, and one
    /// constructed at the scanout format reports BGRA: every CPU consumer of a
    /// IOSurface texture composite Store is declared in guest scanout order, so an RGBA
    /// resident under a scanout-declared mapping costs a whole-frame exchange
    /// per Store.
    ///
    /// `Gva` must answer from its own field and from nothing else. That is the
    /// half a future edit is likely to get wrong in either direction — pinning
    /// it to `false` sends every BGRA-declared render target back through the
    /// blocking readback, and pinning it to `true` silently exchanges R and B
    /// on every RGBA-declared one.
    ///
    /// `Texture` and `Anonymous` must not be, and `Anonymous` in particular is
    /// the pooled path the parity suite uses as its semantic control.
    #[test]
    fn a_targets_order_follows_its_own_namespace() {
        assert!(TargetIdentity::Surface {
            id: 1,
            width: 8,
            height: 8,
            generation: 0,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }
        .is_bgra());
        for (format, bgra) in [
            (translate::pixel::RESIDENT_RGBA_FORMAT, false),
            (translate::pixel::SCANOUT_FORMAT, true),
            (ash::vk::Format::R8G8B8A8_SRGB, false),
            (ash::vk::Format::B8G8R8A8_SRGB, true),
        ] {
            let stored = translate::pixel::storage_format(format);
            let gva = TargetIdentity::Gva {
                gva: 0x1000,
                width: 8,
                height: 8,
                generation: 0,
                format: translate::pixel::texel_layout_of(stored).unwrap(),
            };
            assert_eq!(
                translate::pixel::vk_texel_layout(gva.resident_layout()),
                stored,
                "{gva:?} must answer its key"
            );
            assert_eq!(gva.is_bgra(), bgra, "{gva:?} must answer from its key");
        }
        for other in [
            TargetIdentity::Texture {
                ref_: 2,
                width: 8,
                height: 8,
                generation: 0,
                stencil: false,
            },
            TargetIdentity::Anonymous { slot: 0 },
        ] {
            assert!(!other.is_bgra(), "{other:?} must stay semantic RGBA");
        }
    }

    /// Two allocations at one address declaring different formats are two keys.
    ///
    /// The format has to be *in* the key, not beside it. If it were not, both
    /// would hash to one registry slot whose image can only be built one way,
    /// and `registry_ensure` answers a requested format that disagrees with the
    /// slot's by destroying and recreating the image — every frame, for as long
    /// as both keep drawing.
    ///
    /// The third format here is the point. `R16G16B16A16_SFLOAT` and
    /// `R8G8B8A8_UNORM` are the **same channel order** and different images, so
    /// while this key held a `bgra: bool` they were one entry — and the wider
    /// one could not be asked for at all, which is why nothing noticed. A key
    /// that separates the two orders but not those two formats passes the first
    /// assertion here and fails the second.
    #[test]
    fn a_gva_targets_format_separates_it_from_the_same_address_in_another_format() {
        let at = |format| TargetIdentity::Gva {
            gva: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            format,
        };
        let rgba8 = at(reims_vgpu_protocol::TexelLayout::Rgba8);
        let bgra8 = at(reims_vgpu_protocol::TexelLayout::Bgra8);
        let rgba16f = at(reims_vgpu_protocol::TexelLayout::Rgba16Float);
        assert_ne!(rgba8, bgra8);
        assert_ne!(
            rgba8, rgba16f,
            "two widths of one channel order are two residents"
        );
        let mut seen = std::collections::HashSet::new();
        for (id, what) in [(bgra8, "bgra8"), (rgba8, "rgba8"), (rgba16f, "rgba16f")] {
            assert!(
                seen.insert(id),
                "{what} must not collide in the registry's key space"
            );
        }
    }

    /// The registry question and the conflict question must answer differently,
    /// and only one of them may look at the format.
    ///
    /// Two colour attachments of one pass over one guest span write that span
    /// twice whatever format each declares, so the MRT alias check has to refuse
    /// the pair — while the registry has to keep them apart, because they are
    /// two images. `==` cannot serve both: it either has the format and misses
    /// the conflict, or lacks it and merges two images into one slot.
    ///
    /// This is the pair the old `bgra: bool` key could not express. Both of
    /// these are RGBA-ordered, so it answered `==` for them and the alias check
    /// fired by accident.
    #[test]
    fn one_span_at_two_formats_is_two_registry_keys_and_still_one_conflict() {
        let at = |format| TargetIdentity::Gva {
            gva: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            format,
        };
        let rgba8 = at(reims_vgpu_protocol::TexelLayout::Rgba8);
        let rgba16f = at(reims_vgpu_protocol::TexelLayout::Rgba16Float);
        assert_ne!(rgba8, rgba16f, "two images, so two registry slots");
        assert!(
            rgba8.aliases(&rgba16f),
            "one guest span, so one destination and a refused pass"
        );

        // A different span is neither, and the two namespaces never alias each
        // other however their numbers line up.
        let elsewhere = TargetIdentity::Gva {
            gva: 0x5000,
            width: 64,
            height: 64,
            generation: 7,
            format: reims_vgpu_protocol::TexelLayout::Rgba8,
        };
        assert!(!rgba8.aliases(&elsewhere));
        assert!(!rgba8.aliases(&TargetIdentity::Surface {
            id: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }));
    }
}
