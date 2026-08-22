//! Typed failures while materializing a validated Vulkan compute dispatch.
//!
//! These checks protect persistent storage-image residency. They are later
//! than `ComputeValidationDecline`: the request is structurally valid, but the
//! resident snapshot observed at execution no longer matches what the runtime
//! staged.

use super::types::StorageImageFormat;
use reims_vgpu_core::ComputeStorageResidencyKey;
use reims_vgpu_observe::Decline;

/// A specific failure while preparing a validated compute dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeExecutionDecline {
    NullSampledImageUnsupported {
        binding: u32,
    },
    NullSamplerUnsupported {
        binding: u32,
    },
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
    /// Expected to stay at zero: `runtime::compute_exec` preserves null texture
    /// and sampler bindings explicitly. A firing is a class that projection
    /// does not cover and is worth reading as a real gap.
    UsedBindingAbsentFromLayout {
        binding: u32,
    },
    /// The module reads push constants and the prepared variant names no
    /// kernel-grid range for the layout to expose.
    ///
    /// The translated entry point culls every invocation whose id falls outside
    /// the grid it reads from push-constant storage. With no range declared,
    /// that storage is undefined: it reads zero on the drivers seen here, which
    /// culls **every** invocation, so the dispatch writes nothing and no
    /// counter moves. Refusing by name is the difference between one visibly
    /// declined dispatch and a kernel that silently does nothing forever.
    KernelGridRangeAbsent,
}

impl Decline for ComputeExecutionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NullSampledImageUnsupported { .. } => {
                "vk_compute_exec_null_sampled_image_unsupported"
            }
            Self::NullSamplerUnsupported { .. } => "vk_compute_exec_null_sampler_unsupported",
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
            Self::ResidentSeedGenerationLost { .. } => {
                "vk_compute_exec_resident_seed_generation_lost"
            }
            Self::ResidentAllocatorLiveSlotMissing { .. } => {
                "vk_compute_exec_resident_allocator_live_slot_missing"
            }
            Self::ResidentRekeyWouldDropPinned { .. } => {
                "vk_compute_exec_resident_rekey_would_drop_pinned"
            }
            Self::KernelGridRangeAbsent => "vk_compute_exec_kernel_grid_range_absent",
            Self::UsedBindingAbsentFromLayout { .. } => {
                "vk_compute_exec_used_binding_absent_from_layout"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NullSampledImageUnsupported { binding }
            | Self::NullSamplerUnsupported { binding } => {
                vec![("binding", binding.to_string())]
            }
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
            Self::KernelGridRangeAbsent => Vec::new(),
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
    let mut fields = vec![
        ("residency_width", identity.width.to_string()),
        ("residency_height", identity.height.to_string()),
        ("residency_pixel_format", identity.pixel_format.to_string()),
    ];
    match identity.origin {
        reims_vgpu_core::ComputeStorageOrigin::Surface {
            mapping_id,
            map_generation,
            surface_offset,
            surface_bpr,
            span_end,
        } => fields.extend([
            ("residency_kind", "surface".to_string()),
            ("residency_mapping_id", mapping_id.to_string()),
            ("residency_map_generation", map_generation.to_string()),
            ("residency_surface_offset", format!("{surface_offset:#x}")),
            ("residency_surface_bpr", surface_bpr.to_string()),
            ("residency_span_end", span_end.to_string()),
        ]),
        reims_vgpu_core::ComputeStorageOrigin::Linear {
            resource,
            gva,
            row_stride,
            span_end,
        } => fields.extend([
            ("residency_kind", "linear".to_string()),
            ("residency_resource", resource.index().to_string()),
            (
                "residency_resource_generation",
                resource.generation().to_string(),
            ),
            ("residency_gva", format!("{gva:#x}")),
            ("residency_row_stride", row_stride.to_string()),
            ("residency_span_end", span_end.to_string()),
        ]),
        reims_vgpu_core::ComputeStorageOrigin::Heap { resource } => fields.extend([
            ("residency_kind", "heap".to_string()),
            ("residency_resource", resource.index().to_string()),
            (
                "residency_resource_generation",
                resource.generation().to_string(),
            ),
        ]),
    }
    fields
}

reims_vgpu_observe::decline_display!(ComputeExecutionDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey::surface(7, 8, 0x9000, 256, 4096, 64, 32, 80)
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
        assert_eq!(before, 6, "the compute executor's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate compute-execution slug");
    }

    #[test]
    fn compute_execution_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line =
                reims_vgpu_observe::Emit::decline("compute_execution_test", &decline).render();
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
                ("residency_width", "64".into()),
                ("residency_height", "32".into()),
                ("residency_pixel_format", "80".into()),
                ("residency_kind", "surface".into()),
                ("residency_mapping_id", "7".into()),
                ("residency_map_generation", "8".into()),
                ("residency_surface_offset", "0x9000".into()),
                ("residency_surface_bpr", "256".into()),
                ("residency_span_end", "4096".into()),
            ]
        );
        assert_eq!(
            residency_fields(&ComputeStorageResidencyKey::linear(
                reims_vgpu_protocol::ResourceId::new(11, 4),
                0xa000,
                512,
                8192,
                32,
                16,
                70,
            )),
            vec![
                ("residency_width", "32".into()),
                ("residency_height", "16".into()),
                ("residency_pixel_format", "70".into()),
                ("residency_kind", "linear".into()),
                ("residency_resource", "11".into()),
                ("residency_resource_generation", "4".into()),
                ("residency_gva", "0xa000".into()),
                ("residency_row_stride", "512".into()),
                ("residency_span_end", "8192".into()),
            ]
        );
        assert_eq!(
            residency_fields(&ComputeStorageResidencyKey::heap(
                reims_vgpu_protocol::ResourceId::new(12, 4),
                8,
                4,
                60,
            )),
            vec![
                ("residency_width", "8".into()),
                ("residency_height", "4".into()),
                ("residency_pixel_format", "60".into()),
                ("residency_kind", "heap".into()),
                ("residency_resource", "12".into()),
                ("residency_resource_generation", "4".into()),
            ]
        );
    }
}
