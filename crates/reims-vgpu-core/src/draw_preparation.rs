//! Backend-independent failures while preparing one resolved draw.
//!
//! These checks happen before [`crate::DrawRequest`] validation: resolving the
//! pipeline and its stage libraries, extracting the guest program, translating
//! each stage, and resolving semantic resources. Translation itself remains an
//! executor concern, represented by the generic `TranslationDecline` payload;
//! the surrounding preparation vocabulary is owned here.

use crate::{
    pixel_format::ColorAttachmentDecline, IndexLoadReason, MtlbDecline, PastTableBind,
    SecondaryMrtRefusal, ShaderStage,
};
use reims_vgpu_observe::Decline;
use reims_vgpu_protocol::{
    ObjectKind, PassActionDecodeError, PipelineStateDecodeError, VertexFormatDecodeError,
    VertexStepDecodeError,
};

/// Failure to construct the semantic attachment set for one render pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentTargetRole {
    Source,
    Resolve,
}

impl AttachmentTargetRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Resolve => "resolve",
        }
    }
}

/// The source attempting to provision an already occupied sampler binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerBindingSource {
    Stream,
    Reflected,
}

impl SamplerBindingSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::Reflected => "reflected",
        }
    }
}

/// Failure to construct the semantic attachment set for one render pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentPlanDecline {
    PassAction {
        slot: u32,
        reason: PassActionDecodeError,
    },
    TargetUnresolved {
        slot: u32,
        texture_ref: u32,
        role: AttachmentTargetRole,
    },
    ResolveTargetMismatch {
        slot: u32,
        source_ref: u32,
        resolve_ref: u32,
        source_width: u32,
        source_height: u32,
        resolve_width: u32,
        resolve_height: u32,
        source_format: u16,
        resolve_format: u16,
    },
}

impl AttachmentPlanDecline {
    pub const fn slot(self) -> u32 {
        match self {
            Self::PassAction { slot, .. }
            | Self::TargetUnresolved { slot, .. }
            | Self::ResolveTargetMismatch { slot, .. } => slot,
        }
    }

    /// Stable first-sight identity for this exact refusal subject.
    ///
    /// The class occupies the high byte. The remaining fields distinguish two
    /// failures in one pipeline slot without making observation depend on
    /// formatted decline fields.
    pub fn latch(self) -> u64 {
        let (class, subject) = match self {
            Self::PassAction { reason, .. } => {
                let class = match reason {
                    PassActionDecodeError::Load(_) => 1u64,
                    PassActionDecodeError::Store(_) => 2u64,
                };
                (class, u64::from(reason.raw()))
            }
            Self::TargetUnresolved {
                texture_ref, role, ..
            } => {
                let class = match role {
                    AttachmentTargetRole::Source => 3u64,
                    AttachmentTargetRole::Resolve => 4u64,
                };
                (class, u64::from(texture_ref))
            }
            Self::ResolveTargetMismatch { resolve_ref, .. } => (5, u64::from(resolve_ref)),
        };
        let hash = crate::fnv::fold_u64(crate::fnv::FNV_OFFSET_BASIS, class);
        let hash = crate::fnv::fold_u64(hash, u64::from(self.slot()));
        crate::fnv::fold_u64(hash, subject)
    }
}

impl Decline for AttachmentPlanDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::PassAction {
                reason: PassActionDecodeError::Load(_),
                ..
            } => "draw_prepare_attachment_load_action",
            Self::PassAction {
                reason: PassActionDecodeError::Store(_),
                ..
            } => "draw_prepare_attachment_store_action",
            Self::TargetUnresolved { .. } => "draw_prepare_attachment_target_unresolved",
            Self::ResolveTargetMismatch { .. } => "draw_prepare_attachment_resolve_target_mismatch",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::PassAction { slot, reason } => vec![
                ("slot", slot.to_string()),
                ("value", reason.raw().to_string()),
                ("action_reason", reason.slug().to_string()),
            ],
            Self::TargetUnresolved {
                slot,
                texture_ref,
                role,
            } => vec![
                ("slot", slot.to_string()),
                ("texture_ref", texture_ref.to_string()),
                ("role", role.as_str().to_string()),
            ],
            Self::ResolveTargetMismatch {
                slot,
                source_ref,
                resolve_ref,
                source_width,
                source_height,
                resolve_width,
                resolve_height,
                source_format,
                resolve_format,
            } => vec![
                ("slot", slot.to_string()),
                ("source_ref", source_ref.to_string()),
                ("resolve_ref", resolve_ref.to_string()),
                ("source_width", source_width.to_string()),
                ("source_height", source_height.to_string()),
                ("resolve_width", resolve_width.to_string()),
                ("resolve_height", resolve_height.to_string()),
                ("source_format", format!("{source_format:#x}")),
                ("resolve_format", format!("{resolve_format:#x}")),
            ],
        }
    }
}

/// Two independently selected consumers attempted to own one draw completion.
/// A prepared draw has exactly one route; branch order must never decide which
/// guest-visible Store effect wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletionRouteConflict {
    pub current: &'static str,
    pub requested: &'static str,
}

/// A specific pipeline/stage preparation failure before engine request validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawPreparationDecline<TranslationDecline> {
    PipelineMissing {
        task_id: u32,
        pipeline_ref: u32,
    },
    VertexMtlbMissing {
        task_id: u32,
        function_ref: u32,
    },
    FragmentMtlbMissing {
        task_id: u32,
        function_ref: u32,
    },
    VertexAirExtract {
        function_ref: u32,
        reason: MtlbDecline,
    },
    FragmentAirExtract {
        function_ref: u32,
        reason: MtlbDecline,
    },
    VertexTranslate {
        pipeline_ref: u32,
        reason: TranslationDecline,
    },
    FragmentTranslate {
        pipeline_ref: u32,
        reason: TranslationDecline,
    },
    GeometryUnsupported {
        width: u32,
        height: u32,
    },
    ColorAttachmentFormat {
        reason: ColorAttachmentDecline,
    },
    /// A live bind names a slot past its class's argument table, so no encoder
    /// of the resolved argument interface has anywhere to put it. The whole
    /// draw is refused rather than silently dropping that bind.
    BindSlotPastTable {
        pipeline_ref: u32,
        bind: PastTableBind,
    },
    /// The guest's colour list names more than one render target and one of the
    /// secondary attachments cannot be built, so the whole draw is refused.
    ///
    /// The alternative is what this device used to do: drop every secondary and
    /// execute the draw against slot 0 alone. That writes a frame the guest has
    /// no way to know is wrong — a fragment shader's `location` 1.. outputs go
    /// nowhere and a later pass sampling that attachment reads whatever was
    /// there before. See
    /// [`crate::MrtDrop`] for which checks bail.
    SecondaryTargetUnbuildable {
        pipeline_ref: u32,
        refusal: SecondaryMrtRefusal,
    },
    VertexBufferMissing {
        index: u32,
        buffer_ref: u32,
        offset: u64,
    },
    FragmentBufferMissing {
        index: u32,
        buffer_ref: u32,
        offset: u64,
    },
    VertexAttributeFormat {
        location: u32,
        buffer_index: u32,
        raw_format: u32,
        reason: VertexFormatDecodeError,
    },
    StageInBytesMissing {
        location: u32,
        buffer_index: u32,
        raw_format: u32,
        stride: u32,
    },
    VertexStepFunctionUnsupported {
        location: u32,
        buffer_index: u32,
        reason: VertexStepDecodeError,
    },
    ColorInputMrtUnsupported {
        destination_index: u32,
    },
    AttachmentAliasIdentityMissing {
        index: u32,
        texture_ref: u32,
    },
    TextureResolveMissing {
        stage: &'static str,
        index: u32,
        texture_ref: u32,
        detail: String,
    },
    TextureDimensionUnsupported {
        stage: &'static str,
        index: u32,
        texture_ref: u32,
        binding: u32,
        kind: String,
    },
    /// The render engine currently exposes guest textures as sampled images.
    /// Binding a reflected storage image through that descriptor type would be
    /// invalid Vulkan, so refuse before constructing the request.
    TextureAccessUnsupported {
        stage: &'static str,
        index: u32,
        texture_ref: u32,
        binding: u32,
        access: &'static str,
    },
    /// A valid reflected resource needs a runtime provisioning path this
    /// backend does not implement. Treating its index as an ordinary stream
    /// bind—or as absent—leaves real shader work unbound.
    ReflectedResourceUnsupported {
        stage: &'static str,
        index: u32,
        binding: Option<u32>,
        kind: &'static str,
    },
    ReflectedInterfaceUnsupported {
        stage: &'static str,
        feature: &'static str,
        count: usize,
    },
    IndexLoad {
        reason: IndexLoadReason,
    },
    ChainResidentIdentityMissing {
        target_gva: u64,
        width: u32,
        height: u32,
    },
    DepthStencilStateMissing {
        task_id: u32,
        state_ref: u32,
        detail: &'static str,
    },
    BlendState {
        reason: PipelineStateDecodeError,
    },
    DepthCompare {
        reason: PipelineStateDecodeError,
    },
    StencilState {
        face: &'static str,
        reason: PipelineStateDecodeError,
    },
    MultisampleAttachmentSampleCountMismatch {
        attachment: u32,
        raster: u32,
    },
    MultisampleResolveShapeUnsupported {
        color_targets: u32,
        depth: bool,
        color_input: bool,
    },
    MultisampleStoreActionUnsupported {
        store_action: u16,
    },
    MultisampleLoadActionUnsupported {
        load_action: u16,
    },
    CompletionRouteConflict {
        conflict: CompletionRouteConflict,
    },
    SamplerBindingCollision {
        stage: ShaderStage,
        index: u32,
        binding: u32,
        source: SamplerBindingSource,
    },
    SamplerEntryMissing {
        sampler_ref: u32,
        binding: u32,
    },
    SamplerObjectType {
        sampler_ref: u32,
        binding: u32,
        object_type: ObjectKind,
    },
    SamplerDescriptorMissing {
        sampler_ref: u32,
        binding: u32,
    },
    SamplerDescriptorShort {
        sampler_ref: u32,
        binding: u32,
        descriptor_len: usize,
    },
    SamplerDescriptorUnknownType {
        sampler_ref: u32,
        binding: u32,
        descriptor_len: usize,
        tag: Option<u32>,
    },
    SamplerDescriptorUnsupported {
        sampler_ref: u32,
        binding: u32,
        descriptor_len: usize,
        tag: Option<u32>,
        declared_len: Option<u32>,
    },
    SamplerMinFilterTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerMagFilterTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerMipFilterTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerAddressSTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerAddressTTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerAddressRTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerBorderColorTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    SamplerCompareFunctionTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: PipelineStateDecodeError,
    },
    StaticSamplerReductionUnsupported {
        stage: &'static str,
        binding: u32,
        reduction: String,
        raw_words: [u64; 2],
    },
    StaticSamplerLodBiasUnsupported {
        stage: &'static str,
        binding: u32,
        lod_bias_bits: u32,
        raw_words: [u64; 2],
    },
    StaticSamplerMinFilterUnsupported {
        stage: &'static str,
        binding: u32,
    },
    StaticSamplerMagFilterUnsupported {
        stage: &'static str,
        binding: u32,
    },
}

fn log_token(detail: &str) -> String {
    detail.replace(char::is_whitespace, "_")
}

impl<TranslationDecline: Decline> Decline for DrawPreparationDecline<TranslationDecline> {
    fn slug(&self) -> &'static str {
        match self {
            Self::PipelineMissing { .. } => "draw_prepare_pipeline_missing",
            Self::VertexMtlbMissing { .. } => "draw_prepare_vertex_mtlb_missing",
            Self::FragmentMtlbMissing { .. } => "draw_prepare_fragment_mtlb_missing",
            Self::VertexAirExtract { reason, .. } | Self::FragmentAirExtract { reason, .. } => {
                reason.slug()
            }
            Self::VertexTranslate { reason, .. } | Self::FragmentTranslate { reason, .. } => {
                reason.slug()
            }
            Self::GeometryUnsupported { .. } => "draw_prepare_geometry_unsupported",
            Self::ColorAttachmentFormat { reason } => match reason {
                ColorAttachmentDecline::UnknownPixelFormat { .. } => {
                    "draw_prepare_color_attachment_unknown_pixel_format"
                }
                ColorAttachmentDecline::NoColorAttachmentFormat { .. } => {
                    "draw_prepare_color_attachment_format_unsupported"
                }
            },
            Self::BindSlotPastTable { .. } => "draw_prepare_bind_slot_past_table",
            // One slug for all five `MrtDrop` reasons, with the reason carried
            // as a field. Delegating to `reason.slug()` the way the AIR-extract
            // arms do would make this refusal share `fail_once`'s latch with the
            // `note_secondary_mrt_drop` census that emits the same five slugs,
            // and the census fires first — so the refusal would be silent for
            // exactly the geometry the census had already reported.
            Self::SecondaryTargetUnbuildable { .. } => "draw_prepare_secondary_target_unbuildable",
            Self::VertexBufferMissing { .. } => "draw_prepare_vertex_buffer_missing",
            Self::FragmentBufferMissing { .. } => "draw_prepare_fragment_buffer_missing",
            Self::VertexAttributeFormat { .. } => "draw_prepare_vertex_attribute_format",
            Self::StageInBytesMissing { .. } => "draw_prepare_stage_in_bytes_missing",
            // The one semantic decoder that needs two slugs, and the
            // asymmetry with its nine siblings is the point rather than an
            // oversight.
            //
            // Every other translate entry point returns exactly one reason
            // variant — `filter` only `UnknownSamplerFilter`, `address_mode`
            // only `UnknownSamplerAddressMode`, `attribute_format` only
            // `UnknownVertexFormat` — so for those one fixed string loses
            // nothing and says more, because it also names *which* field
            // failed. `step_function` is the one that returns **two**, and the
            // distinction is the whole reason the second exists: a tessellation
            // step rate is a value this backend recognises and has no Vulkan
            // spelling for, not a value it failed to recognise.
            //
            // Under one fixed string the two rendered identically apart from
            // `value`, which is the ambiguity this typed split exists to end.
            // Delegating to `reason.slug()` the way `VertexTranslate` and
            // `IndexLoad` do is not the repair either: those carry
            // `M2vCacheDecline` and `IndexLoadReason`, whose slugs already
            // begin `m2v_`/`mtlb_`, while the semantic decoder's slugs are
            // unprefixed and would leave the emitted name unattributable to a
            // subsystem.
            Self::VertexStepFunctionUnsupported { reason, .. } => match reason {
                VertexStepDecodeError::TessellationUnsupported(_) => {
                    "draw_prepare_vertex_step_function_per_patch"
                }
                _ => "draw_prepare_vertex_step_function_unsupported",
            },
            Self::ColorInputMrtUnsupported { .. } => "draw_prepare_color_input_mrt_unsupported",
            Self::AttachmentAliasIdentityMissing { .. } => {
                "draw_prepare_attachment_alias_identity_missing"
            }
            Self::TextureResolveMissing { .. } => "draw_prepare_texture_resolve_missing",
            Self::TextureDimensionUnsupported { .. } => {
                "draw_prepare_texture_dimension_unsupported"
            }
            Self::TextureAccessUnsupported { .. } => "draw_prepare_texture_access_unsupported",
            Self::ReflectedResourceUnsupported { .. } => {
                "draw_prepare_reflected_resource_unsupported"
            }
            Self::ReflectedInterfaceUnsupported { .. } => {
                "draw_prepare_reflected_interface_unsupported"
            }
            Self::IndexLoad { reason } => reason.slug(),
            Self::ChainResidentIdentityMissing { .. } => {
                "draw_prepare_chain_resident_identity_missing"
            }
            Self::DepthStencilStateMissing { .. } => "draw_prepare_depth_stencil_state_missing",
            Self::BlendState { .. } => "draw_prepare_blend_state",
            Self::DepthCompare { .. } => "draw_prepare_depth_compare",
            Self::StencilState { .. } => "draw_prepare_stencil_state",
            Self::MultisampleAttachmentSampleCountMismatch { .. } => {
                "multisample_attachment_sample_count_mismatch"
            }
            Self::MultisampleResolveShapeUnsupported { .. } => {
                "multisample_resolve_shape_unsupported"
            }
            Self::MultisampleStoreActionUnsupported { .. } => {
                "multisample_store_action_unsupported"
            }
            Self::MultisampleLoadActionUnsupported { .. } => "multisample_load_action_unsupported",
            Self::CompletionRouteConflict { .. } => "draw_prepare_completion_route_conflict",
            Self::SamplerBindingCollision { .. } => "draw_prepare_sampler_binding_collision",
            Self::SamplerEntryMissing { .. } => "draw_prepare_sampler_no_list_entry",
            Self::SamplerObjectType { .. } => "draw_prepare_sampler_wrong_type",
            Self::SamplerDescriptorMissing { .. } => "draw_prepare_sampler_desc_read",
            Self::SamplerDescriptorShort { .. } => "draw_prepare_sampler_descriptor_short",
            Self::SamplerDescriptorUnknownType { .. } => {
                "draw_prepare_sampler_descriptor_unknown_type"
            }
            Self::SamplerDescriptorUnsupported { .. } => {
                "draw_prepare_sampler_descriptor_unsupported"
            }
            Self::SamplerMinFilterTranslation { .. } => {
                "draw_prepare_sampler_min_filter_translation"
            }
            Self::SamplerMagFilterTranslation { .. } => {
                "draw_prepare_sampler_mag_filter_translation"
            }
            Self::SamplerMipFilterTranslation { .. } => {
                "draw_prepare_sampler_mip_filter_translation"
            }
            Self::SamplerAddressSTranslation { .. } => "draw_prepare_sampler_address_s_translation",
            Self::SamplerAddressTTranslation { .. } => "draw_prepare_sampler_address_t_translation",
            Self::SamplerAddressRTranslation { .. } => "draw_prepare_sampler_address_r_translation",
            Self::SamplerBorderColorTranslation { .. } => {
                "draw_prepare_sampler_border_color_translation"
            }
            Self::SamplerCompareFunctionTranslation { .. } => {
                "draw_prepare_sampler_compare_function_translation"
            }
            Self::StaticSamplerReductionUnsupported { .. } => {
                "draw_prepare_static_sampler_reduction_unsupported"
            }
            Self::StaticSamplerLodBiasUnsupported { .. } => {
                "draw_prepare_static_sampler_lod_bias_unsupported"
            }
            Self::StaticSamplerMinFilterUnsupported { .. } => {
                "draw_prepare_static_sampler_min_filter_unsupported"
            }
            Self::StaticSamplerMagFilterUnsupported { .. } => {
                "draw_prepare_static_sampler_mag_filter_unsupported"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::PipelineMissing {
                task_id,
                pipeline_ref,
            } => vec![
                ("task_id", task_id.to_string()),
                ("pipeline_ref", pipeline_ref.to_string()),
            ],
            Self::SecondaryTargetUnbuildable {
                pipeline_ref,
                refusal,
            } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("slot", refusal.slot.to_string()),
                // The census slug, so one grep finds both the refusal and the
                // `note_secondary_mrt_drop` line that reports the same check.
                ("mrt_reason", refusal.reason.slug().to_string()),
            ],
            Self::BindSlotPastTable { pipeline_ref, bind } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("class", bind.class.name().to_string()),
                ("stage", bind.stage_name().to_string()),
                ("index", bind.index.to_string()),
                ("table", bind.class.table().to_string()),
                ("ref", bind.resource_ref.to_string()),
            ],
            Self::VertexMtlbMissing {
                task_id,
                function_ref,
            }
            | Self::FragmentMtlbMissing {
                task_id,
                function_ref,
            } => vec![
                ("task_id", task_id.to_string()),
                ("function_ref", function_ref.to_string()),
            ],
            Self::VertexAirExtract {
                function_ref,
                reason,
            } => {
                let mut fields = vec![
                    ("stage", "vertex".to_string()),
                    ("function_ref", function_ref.to_string()),
                ];
                fields.extend(reason.fields());
                fields
            }
            Self::FragmentAirExtract {
                function_ref,
                reason,
            } => {
                let mut fields = vec![
                    ("stage", "fragment".to_string()),
                    ("function_ref", function_ref.to_string()),
                ];
                fields.extend(reason.fields());
                fields
            }
            Self::VertexTranslate {
                pipeline_ref,
                reason,
            }
            | Self::FragmentTranslate {
                pipeline_ref,
                reason,
            } => {
                let mut fields = vec![("pipeline_ref", pipeline_ref.to_string())];
                fields.extend(reason.fields());
                fields
            }
            Self::GeometryUnsupported { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
            Self::ColorAttachmentFormat { reason } => reason.fields(),
            Self::VertexBufferMissing {
                index,
                buffer_ref,
                offset,
            }
            | Self::FragmentBufferMissing {
                index,
                buffer_ref,
                offset,
            } => vec![
                ("index", index.to_string()),
                ("buffer_ref", buffer_ref.to_string()),
                ("offset", offset.to_string()),
            ],
            Self::VertexAttributeFormat {
                location,
                buffer_index,
                raw_format,
                reason,
            } => {
                let mut fields = vec![
                    ("location", location.to_string()),
                    ("buffer_index", buffer_index.to_string()),
                    ("raw_format", raw_format.to_string()),
                ];
                fields.push(("value", reason.raw().to_string()));
                fields
            }
            Self::StageInBytesMissing {
                location,
                buffer_index,
                raw_format,
                stride,
            } => vec![
                ("location", location.to_string()),
                ("buffer_index", buffer_index.to_string()),
                ("raw_format", raw_format.to_string()),
                ("stride", stride.to_string()),
            ],
            Self::VertexStepFunctionUnsupported {
                location,
                buffer_index,
                reason,
            } => {
                let mut fields = vec![
                    ("location", location.to_string()),
                    ("buffer_index", buffer_index.to_string()),
                ];
                fields.push(("value", reason.raw().to_string()));
                fields
            }
            Self::ColorInputMrtUnsupported { destination_index } => {
                vec![("destination_index", destination_index.to_string())]
            }
            Self::AttachmentAliasIdentityMissing { index, texture_ref } => vec![
                ("index", index.to_string()),
                ("texture_ref", texture_ref.to_string()),
            ],
            Self::TextureResolveMissing {
                stage,
                index,
                texture_ref,
                detail,
            } => vec![
                ("stage", (*stage).to_string()),
                ("index", index.to_string()),
                ("texture_ref", texture_ref.to_string()),
                ("detail", log_token(detail)),
            ],
            Self::TextureDimensionUnsupported {
                stage,
                index,
                texture_ref,
                binding,
                kind,
            } => vec![
                ("stage", (*stage).to_string()),
                ("index", index.to_string()),
                ("texture_ref", texture_ref.to_string()),
                ("binding", binding.to_string()),
                ("kind", log_token(kind)),
            ],
            Self::TextureAccessUnsupported {
                stage,
                index,
                texture_ref,
                binding,
                access,
            } => vec![
                ("stage", (*stage).to_string()),
                ("index", index.to_string()),
                ("texture_ref", texture_ref.to_string()),
                ("binding", binding.to_string()),
                ("access", (*access).to_string()),
            ],
            Self::ReflectedResourceUnsupported {
                stage,
                index,
                binding,
                kind,
            } => vec![
                ("stage", (*stage).to_string()),
                ("index", index.to_string()),
                (
                    "binding",
                    binding.map_or_else(|| "none".to_string(), |value| value.to_string()),
                ),
                ("kind", (*kind).to_string()),
            ],
            Self::ReflectedInterfaceUnsupported {
                stage,
                feature,
                count,
            } => vec![
                ("stage", (*stage).to_string()),
                ("feature", (*feature).to_string()),
                ("count", count.to_string()),
            ],
            Self::ChainResidentIdentityMissing {
                target_gva,
                width,
                height,
            } => vec![
                ("target_gva", format!("{target_gva:#x}")),
                ("width", width.to_string()),
                ("height", height.to_string()),
            ],
            Self::DepthStencilStateMissing {
                task_id,
                state_ref,
                detail,
            } => vec![
                ("task_id", task_id.to_string()),
                ("state_ref", state_ref.to_string()),
                ("detail", (*detail).to_string()),
            ],
            Self::BlendState { reason } | Self::DepthCompare { reason } => {
                vec![("value", reason.raw().to_string())]
            }
            Self::StencilState { face, reason } => vec![
                ("face", (*face).to_string()),
                ("value", reason.raw().to_string()),
                ("state_reason", reason.slug().to_string()),
            ],
            Self::CompletionRouteConflict { conflict } => vec![
                ("current", conflict.current.to_string()),
                ("requested", conflict.requested.to_string()),
            ],
            Self::MultisampleAttachmentSampleCountMismatch { attachment, raster } => vec![
                ("attachment", attachment.to_string()),
                ("raster", raster.to_string()),
            ],
            Self::MultisampleResolveShapeUnsupported {
                color_targets,
                depth,
                color_input,
            } => vec![
                ("color_targets", color_targets.to_string()),
                ("depth", u8::from(*depth).to_string()),
                ("color_input", u8::from(*color_input).to_string()),
            ],
            Self::MultisampleStoreActionUnsupported { store_action } => {
                vec![("store_action", store_action.to_string())]
            }
            Self::MultisampleLoadActionUnsupported { load_action } => {
                vec![("load_action", load_action.to_string())]
            }
            Self::SamplerBindingCollision {
                stage,
                index,
                binding,
                source,
            } => vec![
                ("stage", stage.name().to_string()),
                ("index", index.to_string()),
                ("binding", binding.to_string()),
                ("source", source.as_str().to_string()),
            ],
            Self::IndexLoad { reason } => reason.fields(),
            Self::SamplerEntryMissing {
                sampler_ref,
                binding,
            }
            | Self::SamplerDescriptorMissing {
                sampler_ref,
                binding,
            } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("binding", binding.to_string()),
            ],
            Self::SamplerDescriptorShort {
                sampler_ref,
                binding,
                descriptor_len,
            } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("binding", binding.to_string()),
                ("descriptor_len", descriptor_len.to_string()),
            ],
            Self::SamplerDescriptorUnknownType {
                sampler_ref,
                binding,
                descriptor_len,
                tag,
            } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("binding", binding.to_string()),
                ("descriptor_len", descriptor_len.to_string()),
                (
                    "tag",
                    tag.map_or_else(|| "none".into(), |value| format!("{value:#x}")),
                ),
            ],
            Self::SamplerDescriptorUnsupported {
                sampler_ref,
                binding,
                descriptor_len,
                tag,
                declared_len,
            } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("binding", binding.to_string()),
                ("descriptor_len", descriptor_len.to_string()),
                (
                    "tag",
                    tag.map_or_else(|| "none".into(), |value| format!("{value:#x}")),
                ),
                (
                    "declared_len",
                    declared_len.map_or_else(|| "none".into(), |value| value.to_string()),
                ),
            ],
            Self::SamplerObjectType {
                sampler_ref,
                binding,
                object_type,
            } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("binding", binding.to_string()),
                ("object_type", object_type.to_string()),
            ],
            Self::SamplerMinFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerMagFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerMipFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerAddressSTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerAddressTTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerAddressRTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerBorderColorTranslation {
                sampler_ref,
                binding,
                reason,
            }
            | Self::SamplerCompareFunctionTranslation {
                sampler_ref,
                binding,
                reason,
            } => {
                let mut fields = vec![
                    ("sampler_ref", sampler_ref.to_string()),
                    ("binding", binding.to_string()),
                ];
                fields.push(("value", reason.raw().to_string()));
                fields
            }
            Self::StaticSamplerReductionUnsupported {
                stage,
                binding,
                reduction,
                raw_words,
            } => vec![
                ("stage", (*stage).to_string()),
                ("binding", binding.to_string()),
                ("reduction", log_token(reduction)),
                ("raw0", format!("{:016x}", raw_words[0])),
                ("raw1", format!("{:016x}", raw_words[1])),
            ],
            Self::StaticSamplerLodBiasUnsupported {
                stage,
                binding,
                lod_bias_bits,
                raw_words,
            } => vec![
                ("stage", (*stage).to_string()),
                ("binding", binding.to_string()),
                ("lod_bias_bits", format!("{lod_bias_bits:#x}")),
                ("raw0", format!("{:016x}", raw_words[0])),
                ("raw1", format!("{:016x}", raw_words[1])),
            ],
            Self::StaticSamplerMinFilterUnsupported { stage, binding }
            | Self::StaticSamplerMagFilterUnsupported { stage, binding } => vec![
                ("stage", (*stage).to_string()),
                ("binding", binding.to_string()),
            ],
        }
    }
}

impl<TranslationDecline: Decline> std::fmt::Display for DrawPreparationDecline<TranslationDecline> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestTranslationDecline {
        Vertex,
        Fragment,
    }

    impl Decline for TestTranslationDecline {
        fn slug(&self) -> &'static str {
            match self {
                Self::Vertex => "m2v_vertex_translate",
                Self::Fragment => "m2v_fragment_translate",
            }
        }
    }

    fn all() -> Vec<DrawPreparationDecline<TestTranslationDecline>> {
        vec![
            DrawPreparationDecline::PipelineMissing {
                task_id: 1,
                pipeline_ref: 2,
            },
            DrawPreparationDecline::VertexMtlbMissing {
                task_id: 1,
                function_ref: 3,
            },
            DrawPreparationDecline::FragmentMtlbMissing {
                task_id: 1,
                function_ref: 4,
            },
            DrawPreparationDecline::VertexAirExtract {
                function_ref: 3,
                reason: MtlbDecline::WrappedAirMissing { data_len: 1 },
            },
            DrawPreparationDecline::FragmentAirExtract {
                function_ref: 4,
                reason: MtlbDecline::WrapperHeaderTruncated {
                    offset: 1,
                    data_len: 2,
                },
            },
            DrawPreparationDecline::VertexTranslate {
                pipeline_ref: 2,
                reason: TestTranslationDecline::Vertex,
            },
            DrawPreparationDecline::FragmentTranslate {
                pipeline_ref: 2,
                reason: TestTranslationDecline::Fragment,
            },
            DrawPreparationDecline::GeometryUnsupported {
                width: 8192,
                height: 4096,
            },
            DrawPreparationDecline::ColorAttachmentFormat {
                reason: ColorAttachmentDecline::UnknownPixelFormat { format: 0xffff },
            },
            DrawPreparationDecline::ColorAttachmentFormat {
                reason: ColorAttachmentDecline::NoColorAttachmentFormat {
                    format: crate::pixel_format::MTL_FORMAT_R32_FLOAT,
                },
            },
            DrawPreparationDecline::SecondaryTargetUnbuildable {
                pipeline_ref: 2,
                refusal: crate::preparation::SecondaryMrtRefusal {
                    slot: 1,
                    reason: crate::preparation::MrtDrop::UnknownFormat,
                },
            },
            DrawPreparationDecline::VertexBufferMissing {
                index: 1,
                buffer_ref: 5,
                offset: 32,
            },
            DrawPreparationDecline::FragmentBufferMissing {
                index: 2,
                buffer_ref: 6,
                offset: 64,
            },
            DrawPreparationDecline::VertexAttributeFormat {
                location: 3,
                buffer_index: 1,
                raw_format: 99,
                reason: VertexFormatDecodeError(99),
            },
            DrawPreparationDecline::StageInBytesMissing {
                location: 3,
                buffer_index: 1,
                raw_format: 30,
                stride: 16,
            },
            DrawPreparationDecline::VertexStepFunctionUnsupported {
                location: 3,
                buffer_index: 1,
                reason: VertexStepDecodeError::Unknown(9),
            },
            // The same variant under its other reason. It is in the census
            // because it is a distinct slug, and it is the entry that fails if
            // the two ever collapse back into one.
            DrawPreparationDecline::VertexStepFunctionUnsupported {
                location: 3,
                buffer_index: 1,
                reason: VertexStepDecodeError::TessellationUnsupported(3),
            },
            DrawPreparationDecline::ColorInputMrtUnsupported {
                destination_index: 1,
            },
            DrawPreparationDecline::AttachmentAliasIdentityMissing {
                index: 2,
                texture_ref: 7,
            },
            DrawPreparationDecline::TextureResolveMissing {
                stage: "fragment",
                index: 2,
                texture_ref: 7,
                detail: "entry=missing backing=none".into(),
            },
            DrawPreparationDecline::TextureDimensionUnsupported {
                stage: "fragment",
                index: 2,
                texture_ref: 7,
                binding: 34,
                kind: "Cube".into(),
            },
            DrawPreparationDecline::TextureAccessUnsupported {
                stage: "fragment",
                index: 2,
                texture_ref: 7,
                binding: 34,
                access: "storage",
            },
            DrawPreparationDecline::ReflectedResourceUnsupported {
                stage: "fragment",
                index: 2,
                binding: Some(34),
                kind: "embedded_texture",
            },
            DrawPreparationDecline::ReflectedInterfaceUnsupported {
                stage: "fragment",
                feature: "fragment_imageblock",
                count: 2,
            },
            DrawPreparationDecline::ChainResidentIdentityMissing {
                target_gva: 0x12000,
                width: 1280,
                height: 720,
            },
            DrawPreparationDecline::DepthStencilStateMissing {
                task_id: 1,
                state_ref: 8,
                detail: "depth_stencil_desc_read",
            },
            DrawPreparationDecline::BlendState {
                reason: PipelineStateDecodeError::BlendFactor(99),
            },
            DrawPreparationDecline::DepthCompare {
                reason: PipelineStateDecodeError::CompareFunction(99),
            },
            DrawPreparationDecline::StencilState {
                face: "front",
                reason: PipelineStateDecodeError::StencilOperation(99),
            },
            DrawPreparationDecline::CompletionRouteConflict {
                conflict: CompletionRouteConflict {
                    current: "resident_chain",
                    requested: "resident_surface_store",
                },
            },
            DrawPreparationDecline::SamplerBindingCollision {
                stage: ShaderStage::Fragment,
                index: 0,
                binding: 64,
                source: SamplerBindingSource::Stream,
            },
            DrawPreparationDecline::SamplerEntryMissing {
                sampler_ref: 5,
                binding: 64,
            },
            DrawPreparationDecline::SamplerObjectType {
                sampler_ref: 5,
                binding: 64,
                object_type: ObjectKind::TextureView,
            },
            DrawPreparationDecline::SamplerDescriptorMissing {
                sampler_ref: 5,
                binding: 64,
            },
            DrawPreparationDecline::SamplerDescriptorShort {
                sampler_ref: 5,
                binding: 64,
                descriptor_len: 3,
            },
            DrawPreparationDecline::SamplerDescriptorUnknownType {
                sampler_ref: 5,
                binding: 64,
                descriptor_len: 32,
                tag: Some(9),
            },
            DrawPreparationDecline::SamplerDescriptorUnsupported {
                sampler_ref: 5,
                binding: 64,
                descriptor_len: 32,
                tag: Some(9),
                declared_len: Some(32),
            },
            DrawPreparationDecline::SamplerMinFilterTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerFilter(9),
            },
            DrawPreparationDecline::SamplerMagFilterTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerFilter(9),
            },
            DrawPreparationDecline::SamplerMipFilterTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerMipFilter(9),
            },
            DrawPreparationDecline::SamplerAddressSTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerAddressMode(9),
            },
            DrawPreparationDecline::SamplerAddressTTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerAddressMode(9),
            },
            DrawPreparationDecline::SamplerAddressRTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerAddressMode(9),
            },
            DrawPreparationDecline::SamplerBorderColorTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::SamplerBorderColor(9),
            },
            DrawPreparationDecline::SamplerCompareFunctionTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: PipelineStateDecodeError::CompareFunction(9),
            },
            DrawPreparationDecline::StaticSamplerReductionUnsupported {
                stage: "fragment",
                binding: 64,
                reduction: "Minimum".into(),
                raw_words: [1, 2],
            },
            DrawPreparationDecline::StaticSamplerLodBiasUnsupported {
                stage: "fragment",
                binding: 64,
                lod_bias_bits: 1.0f32.to_bits(),
                raw_words: [1, 2],
            },
            DrawPreparationDecline::StaticSamplerMinFilterUnsupported {
                stage: "fragment",
                binding: 64,
            },
            DrawPreparationDecline::StaticSamplerMagFilterUnsupported {
                stage: "fragment",
                binding: 64,
            },
        ]
    }

    #[test]
    fn every_draw_preparation_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(
                slug.starts_with("draw_prepare_")
                    || slug.starts_with("m2v_")
                    || slug.starts_with("mtlb_"),
                "{slug}"
            );
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 49, "the draw-preparation reason census moved");
        assert_eq!(before, slugs.len(), "duplicate draw-preparation slug");
    }

    #[test]
    fn index_load_preserves_the_shared_reason_and_fields() {
        use crate::preparation::IndexLoadReason;

        let decline: DrawPreparationDecline<TestTranslationDecline> =
            DrawPreparationDecline::IndexLoad {
                reason: IndexLoadReason::OutOfBounds,
            };
        assert_eq!(decline.slug(), "draw_index_out_of_bounds");
        assert!(decline.fields().is_empty());
    }

    #[test]
    fn draw_preparation_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = reims_vgpu_observe::Emit::decline("draw_prepare_test", &decline).render();
            assert!(line.starts_with(&format!("draw_prepare_test reason={}", decline.slug())));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }
    }

    #[test]
    fn attachment_plan_declines_are_distinct_and_structured() {
        let declines = [
            AttachmentPlanDecline::PassAction {
                slot: 1,
                reason: PassActionDecodeError::Load(9),
            },
            AttachmentPlanDecline::PassAction {
                slot: 1,
                reason: PassActionDecodeError::Store(9),
            },
            AttachmentPlanDecline::TargetUnresolved {
                slot: 1,
                texture_ref: 7,
                role: AttachmentTargetRole::Resolve,
            },
            AttachmentPlanDecline::ResolveTargetMismatch {
                slot: 1,
                source_ref: 7,
                resolve_ref: 8,
                source_width: 8,
                source_height: 8,
                resolve_width: 4,
                resolve_height: 8,
                source_format: 0x50,
                resolve_format: 0x51,
            },
        ];
        let mut slugs: Vec<_> = declines.iter().map(Decline::slug).collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), declines.len());
        for decline in declines {
            assert!(!decline.fields().is_empty());
        }
    }
}
