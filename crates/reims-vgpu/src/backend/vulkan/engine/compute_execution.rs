//! Typed failures while materializing a validated Vulkan compute dispatch.
//!
//! These checks protect persistent storage-image residency. They are later
//! than `ComputeValidationDecline`: the request is structurally valid, but the
//! resident snapshot observed at execution no longer matches what the runtime
//! staged.

use super::types::{ResidentReclaim, StorageImageFormat, TargetIdentity};
use crate::model::ComputeStorageResidencyKey;
use crate::observe::Decline;

/// A specific failure while preparing a validated compute dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeExecutionDecline {
    ResidentSampleAbsent {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        width: u32,
        height: u32,
    },
    ResidentSampleGenerationMismatch {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        actual_generation: u32,
        expected_generation: u32,
    },
    ResidentSampleByteShapeMismatch {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        source_width: u32,
        source_height: u32,
        source_format: StorageImageFormat,
        source_row_bytes: u64,
        resource_width: u32,
        resource_height: u32,
        resource_format: StorageImageFormat,
        resource_row_bytes: u64,
    },
    SeedSkippedWithoutResidency {
        binding: u32,
        width: u32,
        height: u32,
    },
    /// A sampled binding named both a resident source and more than one mip
    /// level. A resident is one window at one level, so the copy could only
    /// fill the base and every level above it would sample as unwritten.
    ResidentSampleIsNotAPyramid { binding: u32, mip_levels: u32 },
    /// A sampled binding's geometry and level count admit no packed pyramid
    /// layout — a zero extent, a zero texel size, or an overflow. The upload
    /// bytes cannot be apportioned to levels, so nothing is copied.
    SampledPyramidLayout {
        binding: u32,
        width: u32,
        height: u32,
        mip_levels: u32,
    },
    ResidentSeedGenerationLost {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        expected_generation: u32,
    },
    ResidentAllocatorLiveSlotMissing {
        identity: ComputeStorageResidencyKey,
        width: u32,
        height: u32,
        format: StorageImageFormat,
    },
    /// A dispatch asked for this identity at a different image shape while the
    /// resident holding it still owes a deferred writeback.
    ///
    /// One identity maps to one slot, so re-keying it means destroying the old
    /// image — and a pinned resident's pixels exist only there, so that destroys
    /// guest output that was accepted and never landed. Refusing is the faithful
    /// half of the same choice every other removal in this registry already
    /// makes by skipping pinned entries: refuse the request that cannot be
    /// served, rather than serve it by discarding an earlier one.
    ///
    /// Self-clearing rather than terminal. The pin is dropped when the writeback
    /// lands, and the next dispatch bearing this identity re-keys normally, so a
    /// firing means the guest re-shaped a surface between a Store and its flush.
    ResidentRekeyWouldDropPinned {
        identity: ComputeStorageResidencyKey,
        held_width: u32,
        held_height: u32,
        held_format: StorageImageFormat,
        wanted_width: u32,
        wanted_height: u32,
        wanted_format: StorageImageFormat,
    },
    /// The module statically uses a `set = 0` binding the descriptor set layout
    /// this dispatch built does not contain.
    ///
    /// A specification violation — Vulkan requires the pipeline layout to
    /// describe every resource the shader statically uses — and the one layout
    /// defect that cannot be reported from any later point. Mesa's Intel driver
    /// scores each used binding as `(use_count << 7) / array_size` over an array
    /// it sized to `max_binding + 1` and zero-filled, so the absent binding
    /// divides by zero and `vkCreateComputePipelines` kills the process with
    /// `SIGFPE` instead of returning. Refusing one dispatch is the only outcome
    /// left that keeps the VM alive and says why.
    ///
    /// Expected to stay at zero: `runtime::compute_exec` provisions a neutral
    /// sampler and a neutral sampled image for every binding of those classes
    /// the guest left empty, so a firing is a class those passes do not cover
    /// and is worth reading as a real gap.
    UsedBindingAbsentFromLayout { binding: u32 },
    /// A kernel declared `texture2d_ms` at this binding and no retained target
    /// answers to the identity the runtime resolved for it.
    ///
    /// There is no fallback, and that is the contract rather than a gap in this
    /// rail: a multisample image cannot be uploaded from bytes or copied into,
    /// so the target that rendered those samples is the only thing that can
    /// serve the bind. `prior` separates a resident this device reclaimed from
    /// one that was never created — opposite defects.
    MultisampleSampleAbsent {
        binding: u32,
        identity: TargetIdentity,
        prior: Option<ResidentReclaim>,
    },
    /// A retained target answers to the identity but cannot serve this bind.
    ///
    /// Three distinct losses under one name because they are one question asked
    /// of one resident, and the fields say which: nothing has been rendered into
    /// it yet, its extent is not the extent the bind names, or it is
    /// single-sample and binding it to a multisampled shader image would be a
    /// descriptor-type mismatch.
    MultisampleSampleUnusable {
        binding: u32,
        identity: TargetIdentity,
        content_ready: bool,
        resident_width: u32,
        resident_height: u32,
        resident_samples: u32,
        resource_width: u32,
        resource_height: u32,
    },
}

impl Decline for ComputeExecutionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ResidentSampleAbsent { .. } => "vk_compute_exec_resident_sample_absent",
            Self::ResidentSampleGenerationMismatch { .. } => {
                "vk_compute_exec_resident_sample_generation_mismatch"
            }
            Self::ResidentSampleByteShapeMismatch { .. } => {
                "vk_compute_exec_resident_sample_byte_shape_mismatch"
            }
            Self::SeedSkippedWithoutResidency { .. } => {
                "vk_compute_exec_seed_skipped_without_residency"
            }
            Self::ResidentSampleIsNotAPyramid { .. } => {
                "vk_compute_exec_resident_sample_is_not_a_pyramid"
            }
            Self::SampledPyramidLayout { .. } => "vk_compute_exec_sampled_pyramid_layout",
            Self::ResidentSeedGenerationLost { .. } => {
                "vk_compute_exec_resident_seed_generation_lost"
            }
            Self::ResidentAllocatorLiveSlotMissing { .. } => {
                "vk_compute_exec_resident_allocator_live_slot_missing"
            }
            Self::ResidentRekeyWouldDropPinned { .. } => {
                "vk_compute_exec_resident_rekey_would_drop_pinned"
            }
            Self::UsedBindingAbsentFromLayout { .. } => {
                "vk_compute_exec_used_binding_absent_from_layout"
            }
            Self::MultisampleSampleAbsent { .. } => "vk_compute_exec_multisample_sample_absent",
            Self::MultisampleSampleUnusable { .. } => "vk_compute_exec_multisample_sample_unusable",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ResidentSampleAbsent {
                binding,
                identity,
                width,
                height,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.extend([
                    ("resource_width", width.to_string()),
                    ("resource_height", height.to_string()),
                ]);
                fields
            }
            Self::ResidentSampleGenerationMismatch {
                binding,
                identity,
                actual_generation,
                expected_generation,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.extend([
                    ("actual_generation", actual_generation.to_string()),
                    ("expected_generation", expected_generation.to_string()),
                ]);
                fields
            }
            Self::ResidentSampleByteShapeMismatch {
                binding,
                identity,
                source_width,
                source_height,
                source_format,
                source_row_bytes,
                resource_width,
                resource_height,
                resource_format,
                resource_row_bytes,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.extend([
                    ("source_width", source_width.to_string()),
                    ("source_height", source_height.to_string()),
                    ("source_format", format!("{source_format:?}")),
                    ("source_row_bytes", source_row_bytes.to_string()),
                    ("resource_width", resource_width.to_string()),
                    ("resource_height", resource_height.to_string()),
                    ("resource_format", format!("{resource_format:?}")),
                    ("resource_row_bytes", resource_row_bytes.to_string()),
                ]);
                fields
            }
            Self::SeedSkippedWithoutResidency {
                binding,
                width,
                height,
            } => vec![
                ("binding", binding.to_string()),
                ("resource_width", width.to_string()),
                ("resource_height", height.to_string()),
            ],
            Self::ResidentSampleIsNotAPyramid {
                binding,
                mip_levels,
            } => vec![
                ("binding", binding.to_string()),
                ("mip_levels", mip_levels.to_string()),
            ],
            Self::SampledPyramidLayout {
                binding,
                width,
                height,
                mip_levels,
            } => vec![
                ("binding", binding.to_string()),
                ("resource_width", width.to_string()),
                ("resource_height", height.to_string()),
                ("mip_levels", mip_levels.to_string()),
            ],
            Self::ResidentSeedGenerationLost {
                binding,
                identity,
                expected_generation,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.push(("expected_generation", expected_generation.to_string()));
                fields
            }
            Self::ResidentAllocatorLiveSlotMissing {
                identity,
                width,
                height,
                format,
            } => {
                let mut fields = residency_fields(identity);
                fields.extend([
                    ("resource_width", width.to_string()),
                    ("resource_height", height.to_string()),
                    ("format", format!("{format:?}")),
                ]);
                fields
            }
            Self::ResidentRekeyWouldDropPinned {
                identity,
                held_width,
                held_height,
                held_format,
                wanted_width,
                wanted_height,
                wanted_format,
            } => {
                let mut fields = residency_fields(identity);
                fields.extend([
                    ("held_width", held_width.to_string()),
                    ("held_height", held_height.to_string()),
                    ("held_format", format!("{held_format:?}")),
                    ("wanted_width", wanted_width.to_string()),
                    ("wanted_height", wanted_height.to_string()),
                    ("wanted_format", format!("{wanted_format:?}")),
                ]);
                fields
            }
            Self::UsedBindingAbsentFromLayout { binding } => {
                vec![("binding", binding.to_string())]
            }
            Self::MultisampleSampleAbsent {
                binding,
                identity,
                prior,
            } => {
                // Through the shared namespace renderer, not `{identity:?}`: a
                // debug rendering carries spaces, and every field on this
                // channel has to be one token.
                let mut fields = vec![("binding", binding.to_string())];
                fields.extend(super::draw_execution::identity_fields(identity));
                fields.push((
                    "prior",
                    prior.map_or_else(|| "none".to_string(), |p| p.slug().to_string()),
                ));
                fields
            }
            Self::MultisampleSampleUnusable {
                binding,
                identity,
                content_ready,
                resident_width,
                resident_height,
                resident_samples,
                resource_width,
                resource_height,
            } => {
                let mut fields = vec![("binding", binding.to_string())];
                fields.extend(super::draw_execution::identity_fields(identity));
                fields.extend([
                    ("content_ready", u8::from(*content_ready).to_string()),
                    ("resident_width", resident_width.to_string()),
                    ("resident_height", resident_height.to_string()),
                    ("resident_samples", resident_samples.to_string()),
                    ("resource_width", resource_width.to_string()),
                    ("resource_height", resource_height.to_string()),
                ]);
                fields
            }
        }
    }
}

fn binding_identity_fields(
    binding: u32,
    identity: &ComputeStorageResidencyKey,
) -> Vec<(&'static str, String)> {
    let mut fields = vec![("binding", binding.to_string())];
    fields.extend(residency_fields(identity));
    fields
}

pub(super) fn residency_fields(
    identity: &ComputeStorageResidencyKey,
) -> Vec<(&'static str, String)> {
    vec![
        ("residency_mapping_id", identity.mapping_id.to_string()),
        (
            "residency_map_generation",
            identity.map_generation.to_string(),
        ),
        (
            "residency_surface_offset",
            format!("{:#x}", identity.surface_offset),
        ),
        ("residency_surface_bpr", identity.surface_bpr.to_string()),
        ("residency_span_end", identity.span_end.to_string()),
        ("residency_width", identity.width.to_string()),
        ("residency_height", identity.height.to_string()),
        ("residency_pixel_format", identity.pixel_format.to_string()),
        ("residency_texture_ref", identity.texture_ref.to_string()),
    ]
}

crate::observe::decline_display!(ComputeExecutionDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id: 7,
            map_generation: 8,
            surface_offset: 0x9000,
            surface_bpr: 256,
            span_end: 4096,
            width: 64,
            height: 32,
            pixel_format: 80,
            texture_ref: 11,
        }
    }

    fn target_identity() -> TargetIdentity {
        TargetIdentity::Gva {
            gva: 0x350000,
            width: 8,
            height: 8,
            generation: 241,
            format: ash::vk::Format::R8G8B8A8_UNORM,
        }
    }

    fn all() -> Vec<ComputeExecutionDecline> {
        vec![
            ComputeExecutionDecline::ResidentSampleAbsent {
                binding: 32,
                identity: identity(),
                width: 64,
                height: 32,
            },
            ComputeExecutionDecline::ResidentSampleGenerationMismatch {
                binding: 32,
                identity: identity(),
                actual_generation: 8,
                expected_generation: 9,
            },
            ComputeExecutionDecline::ResidentSampleByteShapeMismatch {
                binding: 32,
                identity: identity(),
                source_width: 64,
                source_height: 32,
                source_format: StorageImageFormat::Rgba8Unorm,
                source_row_bytes: 256,
                resource_width: 32,
                resource_height: 32,
                resource_format: StorageImageFormat::Rgba8Unorm,
                resource_row_bytes: 128,
            },
            ComputeExecutionDecline::SeedSkippedWithoutResidency {
                binding: 34,
                width: 64,
                height: 32,
            },
            ComputeExecutionDecline::ResidentSampleIsNotAPyramid {
                binding: 34,
                mip_levels: 7,
            },
            ComputeExecutionDecline::SampledPyramidLayout {
                binding: 34,
                width: 64,
                height: 32,
                mip_levels: 7,
            },
            ComputeExecutionDecline::ResidentSeedGenerationLost {
                binding: 34,
                identity: identity(),
                expected_generation: 9,
            },
            ComputeExecutionDecline::ResidentAllocatorLiveSlotMissing {
                identity: identity(),
                width: 64,
                height: 32,
                format: StorageImageFormat::Rgba8Unorm,
            },
            ComputeExecutionDecline::ResidentRekeyWouldDropPinned {
                identity: identity(),
                held_width: 64,
                held_height: 32,
                held_format: StorageImageFormat::Rgba8Unorm,
                wanted_width: 32,
                wanted_height: 32,
                wanted_format: StorageImageFormat::Rgba8Unorm,
            },
            ComputeExecutionDecline::UsedBindingAbsentFromLayout { binding: 35 },
            ComputeExecutionDecline::MultisampleSampleAbsent {
                binding: 32,
                identity: target_identity(),
                prior: None,
            },
            ComputeExecutionDecline::MultisampleSampleUnusable {
                binding: 32,
                identity: target_identity(),
                content_ready: true,
                resident_width: 8,
                resident_height: 8,
                resident_samples: 1,
                resource_width: 8,
                resource_height: 8,
            },
        ]
    }

    #[test]
    fn every_compute_execution_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_compute_exec_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        // Down from 23: twelve `direct_writeback_*` checks went out with the
        // GPU-direct compute writeback. Every one of them validated the shape
        // of a caller-supplied guest window the dispatch would DMA into —
        // alignment, row stride, offset, overflow, window length — and none of
        // them has anything left to validate now that the copy always lands in
        // a pooled readback the runtime owns. Five more went with the
        // non-2D image shape: the compute rail stages one flat plane window
        // per binding and has no slice or depth axis to refuse.
        //
        // Back up to 8 with the sampled mip pyramid: a resident source cannot
        // answer for a multi-level binding, and a geometry that admits no
        // packed level layout has no way to apportion its upload.
        //
        // 12 now, and this list is complete for the first time. Two variants
        // -- `ResidentRekeyWouldDropPinned` and `UsedBindingAbsentFromLayout`
        // -- had never been in it, so the count they were compared against was
        // never the enum's own size; they are here so that it is. The other two
        // are the multisample pair: a `texture2d_ms` binding whose retained
        // target is gone, and one whose target cannot serve it.
        assert_eq!(before, 12, "the compute executor's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate compute-execution slug");
    }

    #[test]
    fn compute_execution_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("compute_execution_test", &decline).render();
            assert!(line.starts_with(&format!("compute_execution_test reason={}", decline.slug())));
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
    fn residency_fields_preserve_every_identity_component() {
        assert_eq!(
            residency_fields(&identity()),
            vec![
                ("residency_mapping_id", "7".into()),
                ("residency_map_generation", "8".into()),
                ("residency_surface_offset", "0x9000".into()),
                ("residency_surface_bpr", "256".into()),
                ("residency_span_end", "4096".into()),
                ("residency_width", "64".into()),
                ("residency_height", "32".into()),
                ("residency_pixel_format", "80".into()),
                ("residency_texture_ref", "11".into()),
            ]
        );
    }
}
