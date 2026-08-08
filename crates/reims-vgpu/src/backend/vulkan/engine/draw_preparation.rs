//! Typed failures while the runtime prepares one Metal draw for the Vulkan engine.
//!
//! These checks happen before [`super::DrawRequest`] validation: resolving the
//! pipeline and its stage libraries, extracting AIR, and translating each stage.
//! They therefore do not belong to
//! [`super::draw_validation::DrawValidationDecline`], which owns
//! invariants of an already-built engine request.

use crate::backend::vulkan::translate::TranslateReason;
use crate::observe::Decline;
use crate::runtime::draw::IndexLoadReason;
use crate::runtime::m2v_cache::M2vCacheDecline;
use crate::runtime::mtlb::MtlbDecline;

/// A specific pipeline/stage preparation failure before engine request validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawPreparationDecline {
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
        reason: M2vCacheDecline,
    },
    FragmentTranslate {
        pipeline_ref: u32,
        reason: M2vCacheDecline,
    },
    GeometryUnsupported {
        width: u32,
        height: u32,
    },
    /// A live bind names a slot past its class's argument table, so no encoder
    /// of this backend has anywhere to put it. See
    /// [`crate::runtime::draw::first_bind_past_table`] for why the whole draw is
    /// refused rather than the one bind dropped.
    BindSlotPastTable {
        pipeline_ref: u32,
        bind: crate::runtime::draw::PastTableBind,
    },
    /// The guest's colour list names more than one render target and one of the
    /// secondary attachments cannot be built, so the whole draw is refused.
    ///
    /// The alternative is what this device used to do: drop every secondary and
    /// execute the draw against slot 0 alone. That writes a frame the guest has
    /// no way to know is wrong — a fragment shader's `location` 1.. outputs go
    /// nowhere and a later pass sampling that attachment reads whatever was
    /// there before. See
    /// [`crate::runtime::census::present_proxy::MrtDrop`] for which checks bail
    /// and why the Metal arm is the one that settled it.
    SecondaryTargetUnbuildable {
        pipeline_ref: u32,
        refusal: crate::runtime::census::present_proxy::SecondaryMrtRefusal,
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
        reason: TranslateReason,
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
        reason: TranslateReason,
    },
    ColorInputMrtUnsupported {
        destination_index: u32,
    },
    AttachmentAliasIdentityMissing {
        index: u32,
        texture_ref: u32,
    },
    AttachmentAliasResidentNotReady {
        index: u32,
        texture_ref: u32,
        width: u32,
        height: u32,
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
    ChainResidentNotReady {
        target_gva: u64,
        width: u32,
        height: u32,
    },
    IndexLoad {
        reason: IndexLoadReason,
    },
    ChainResidentIdentityMissing {
        target_gva: u64,
        width: u32,
        height: u32,
    },
    SamplerEntryMissing {
        sampler_ref: u32,
        binding: u32,
    },
    SamplerObjectType {
        sampler_ref: u32,
        binding: u32,
        object_type: u8,
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
        reason: TranslateReason,
    },
    SamplerMagFilterTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
    },
    SamplerMipFilterTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
    },
    SamplerAddressSTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
    },
    SamplerAddressTTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
    },
    SamplerAddressRTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
    },
    SamplerBorderColorTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
    },
    SamplerCompareFunctionTranslation {
        sampler_ref: u32,
        binding: u32,
        reason: TranslateReason,
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
    StaticSamplerReflectionDescriptorMissing {
        stage: &'static str,
    },
    StaticSamplerReflectionStateMissing {
        stage: &'static str,
        binding: u32,
    },
}

fn log_token(detail: &str) -> String {
    detail.replace(char::is_whitespace, "_")
}

impl Decline for DrawPreparationDecline {
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
            // The one `TranslateReason` carrier that needs two slugs, and the
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
            // `value`, which is the state `translate::reason` says the split
            // was introduced to end — so that fix never reached the log.
            // Delegating to `reason.slug()` the way `VertexTranslate` and
            // `IndexLoad` do is not the repair either: those carry
            // `M2vCacheDecline` and `IndexLoadReason`, whose slugs already
            // begin `m2v_`/`mtlb_`, while `TranslateReason`'s are unprefixed
            // and would leave the emitted name unattributable to a subsystem.
            Self::VertexStepFunctionUnsupported { reason, .. } => match reason {
                TranslateReason::VertexStepFunctionPerPatch(_) => {
                    "draw_prepare_vertex_step_function_per_patch"
                }
                _ => "draw_prepare_vertex_step_function_unsupported",
            },
            Self::ColorInputMrtUnsupported { .. } => "draw_prepare_color_input_mrt_unsupported",
            Self::AttachmentAliasIdentityMissing { .. } => {
                "draw_prepare_attachment_alias_identity_missing"
            }
            Self::AttachmentAliasResidentNotReady { .. } => {
                "draw_prepare_attachment_alias_resident_not_ready"
            }
            Self::TextureResolveMissing { .. } => "draw_prepare_texture_resolve_missing",
            Self::TextureDimensionUnsupported { .. } => {
                "draw_prepare_texture_dimension_unsupported"
            }
            Self::ChainResidentNotReady { .. } => "draw_prepare_chain_resident_not_ready",
            Self::IndexLoad { reason } => reason.slug(),
            Self::ChainResidentIdentityMissing { .. } => {
                "draw_prepare_chain_resident_identity_missing"
            }
            Self::SamplerEntryMissing { .. } => {
                crate::observe::ladder_slug!("draw_prepare_sampler", no_list_entry)
            }
            Self::SamplerObjectType { .. } => {
                crate::observe::ladder_slug!("draw_prepare_sampler", wrong_type)
            }
            Self::SamplerDescriptorMissing { .. } => {
                crate::observe::ladder_slug!("draw_prepare_sampler", desc_read)
            }
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
            Self::StaticSamplerReflectionDescriptorMissing { .. } => {
                "draw_prepare_static_sampler_reflection_descriptor_missing"
            }
            Self::StaticSamplerReflectionStateMissing { .. } => {
                "draw_prepare_static_sampler_reflection_state_missing"
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
                fields.extend(reason.fields());
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
                fields.extend(reason.fields());
                fields
            }
            Self::ColorInputMrtUnsupported { destination_index } => {
                vec![("destination_index", destination_index.to_string())]
            }
            Self::AttachmentAliasIdentityMissing { index, texture_ref } => vec![
                ("index", index.to_string()),
                ("texture_ref", texture_ref.to_string()),
            ],
            Self::AttachmentAliasResidentNotReady {
                index,
                texture_ref,
                width,
                height,
            } => vec![
                ("index", index.to_string()),
                ("texture_ref", texture_ref.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
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
            Self::ChainResidentNotReady {
                target_gva,
                width,
                height,
            }
            | Self::ChainResidentIdentityMissing {
                target_gva,
                width,
                height,
            } => vec![
                ("target_gva", format!("{target_gva:#x}")),
                ("width", width.to_string()),
                ("height", height.to_string()),
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
                fields.extend(reason.fields());
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
            Self::StaticSamplerReflectionDescriptorMissing { stage } => {
                vec![("stage", (*stage).to_string())]
            }
            Self::StaticSamplerReflectionStateMissing { stage, binding } => vec![
                ("stage", (*stage).to_string()),
                ("binding", binding.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(DrawPreparationDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<DrawPreparationDecline> {
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
                reason: M2vCacheDecline::VertexTranslate {
                    detail: "vertex translator failed".into(),
                },
            },
            DrawPreparationDecline::FragmentTranslate {
                pipeline_ref: 2,
                reason: M2vCacheDecline::FragmentTranslate {
                    detail: "fragment translator failed".into(),
                },
            },
            DrawPreparationDecline::GeometryUnsupported {
                width: 8192,
                height: 4096,
            },
            DrawPreparationDecline::SecondaryTargetUnbuildable {
                pipeline_ref: 2,
                refusal: crate::runtime::census::present_proxy::SecondaryMrtRefusal {
                    slot: 1,
                    reason: crate::runtime::census::present_proxy::MrtDrop::GeometryMismatch,
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
                reason: TranslateReason::UnknownVertexFormat(99),
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
                reason: TranslateReason::UnknownVertexStepFunction(9),
            },
            // The same variant under its other reason. It is in the census
            // because it is a distinct slug, and it is the entry that fails if
            // the two ever collapse back into one.
            DrawPreparationDecline::VertexStepFunctionUnsupported {
                location: 3,
                buffer_index: 1,
                reason: TranslateReason::VertexStepFunctionPerPatch(3),
            },
            DrawPreparationDecline::ColorInputMrtUnsupported {
                destination_index: 1,
            },
            DrawPreparationDecline::AttachmentAliasIdentityMissing {
                index: 2,
                texture_ref: 7,
            },
            DrawPreparationDecline::AttachmentAliasResidentNotReady {
                index: 2,
                texture_ref: 7,
                width: 1280,
                height: 720,
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
            DrawPreparationDecline::ChainResidentNotReady {
                target_gva: 0x12000,
                width: 1280,
                height: 720,
            },
            DrawPreparationDecline::ChainResidentIdentityMissing {
                target_gva: 0x12000,
                width: 1280,
                height: 720,
            },
            DrawPreparationDecline::SamplerEntryMissing {
                sampler_ref: 5,
                binding: 64,
            },
            DrawPreparationDecline::SamplerObjectType {
                sampler_ref: 5,
                binding: 64,
                object_type: 8,
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
                reason: TranslateReason::UnknownSamplerFilter(9),
            },
            DrawPreparationDecline::SamplerMagFilterTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownSamplerFilter(9),
            },
            DrawPreparationDecline::SamplerMipFilterTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownSamplerMipFilter(9),
            },
            DrawPreparationDecline::SamplerAddressSTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownSamplerAddressMode(9),
            },
            DrawPreparationDecline::SamplerAddressTTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownSamplerAddressMode(9),
            },
            DrawPreparationDecline::SamplerAddressRTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownSamplerAddressMode(9),
            },
            DrawPreparationDecline::SamplerBorderColorTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownSamplerBorderColor(9),
            },
            DrawPreparationDecline::SamplerCompareFunctionTranslation {
                sampler_ref: 5,
                binding: 64,
                reason: TranslateReason::UnknownCompareFunction(9),
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
            DrawPreparationDecline::StaticSamplerReflectionDescriptorMissing { stage: "fragment" },
            DrawPreparationDecline::StaticSamplerReflectionStateMissing {
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
        assert_eq!(before, 42, "the draw-preparation reason census moved");
        assert_eq!(before, slugs.len(), "duplicate draw-preparation slug");
    }

    #[test]
    fn index_load_preserves_the_shared_reason_and_fields() {
        use crate::runtime::draw::IndexLoadReason;

        let decline = DrawPreparationDecline::IndexLoad {
            reason: IndexLoadReason::OutOfBounds,
        };
        assert_eq!(decline.slug(), "draw_index_out_of_bounds");
        assert!(decline.fields().is_empty());
    }

    #[test]
    fn draw_preparation_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("draw_prepare_test", &decline).render();
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
}
