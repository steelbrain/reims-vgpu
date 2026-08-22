//! Typed validation failures for a Vulkan compute request.
//!
//! Validation runs before context creation or GPU work. The old rail collapsed
//! seventeen request invariants into `DrawError::Invalid(String)`, including
//! four descriptor-role checks with identical prose.

use reims_vgpu_observe::Decline;

/// A specific malformed or internally inconsistent compute request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeValidationDecline {
    MissingProgram,
    ProgramUnavailable {
        id: u64,
    },
    EmptyEntry,
    EntryInteriorNul,
    ZeroGrid {
        grid: [u32; 3],
    },
    DuplicateStorageBufferBinding {
        binding: u32,
    },
    EmptyStorageBuffer {
        binding: u32,
    },
    DuplicateSampledImageBinding {
        binding: u32,
    },
    SampledArrayElementOutOfRange {
        binding: u32,
        element: u32,
        count: u32,
    },
    SampledZeroGeometry {
        binding: u32,
        width: u32,
        height: u32,
    },
    SampledBytesLength {
        binding: u32,
        actual: usize,
        expected: usize,
    },
    InvalidSamplerLod {
        binding: u32,
        lod_min_bits: u32,
        lod_max_bits: u32,
    },
    DuplicateSamplerBinding {
        binding: u32,
    },
    DuplicateStorageImageBinding {
        binding: u32,
    },
    StorageArrayElementOutOfRange {
        binding: u32,
        element: u32,
        count: u32,
    },
    StorageZeroGeometry {
        binding: u32,
        width: u32,
        height: u32,
    },
    StorageBytesLength {
        binding: u32,
        actual: usize,
        expected: usize,
    },
}

impl Decline for ComputeValidationDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MissingProgram => "vk_compute_validate_missing_program",
            Self::ProgramUnavailable { .. } => "vk_compute_validate_program_unavailable",
            Self::EmptyEntry => "vk_compute_validate_empty_entry",
            Self::EntryInteriorNul => "vk_compute_validate_entry_interior_nul",
            Self::ZeroGrid { .. } => "vk_compute_validate_zero_grid",
            Self::DuplicateStorageBufferBinding { .. } => {
                "vk_compute_validate_duplicate_storage_buffer_binding"
            }
            Self::EmptyStorageBuffer { .. } => "vk_compute_validate_empty_storage_buffer",
            Self::DuplicateSampledImageBinding { .. } => {
                "vk_compute_validate_duplicate_sampled_image_binding"
            }
            Self::SampledArrayElementOutOfRange { .. } => {
                "vk_compute_validate_sampled_array_element_out_of_range"
            }
            Self::SampledZeroGeometry { .. } => "vk_compute_validate_sampled_zero_geometry",
            Self::SampledBytesLength { .. } => "vk_compute_validate_sampled_bytes_length",
            Self::InvalidSamplerLod { .. } => "vk_compute_validate_invalid_sampler_lod",
            Self::DuplicateSamplerBinding { .. } => "vk_compute_validate_duplicate_sampler_binding",
            Self::DuplicateStorageImageBinding { .. } => {
                "vk_compute_validate_duplicate_storage_image_binding"
            }
            Self::StorageArrayElementOutOfRange { .. } => {
                "vk_compute_validate_storage_array_element_out_of_range"
            }
            Self::StorageZeroGeometry { .. } => "vk_compute_validate_storage_zero_geometry",
            Self::StorageBytesLength { .. } => "vk_compute_validate_storage_bytes_length",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ZeroGrid { grid } => vec![
                ("grid_x", grid[0].to_string()),
                ("grid_y", grid[1].to_string()),
                ("grid_z", grid[2].to_string()),
            ],
            Self::DuplicateStorageBufferBinding { binding }
            | Self::EmptyStorageBuffer { binding }
            | Self::DuplicateSampledImageBinding { binding }
            | Self::DuplicateSamplerBinding { binding }
            | Self::DuplicateStorageImageBinding { binding } => {
                vec![("binding", binding.to_string())]
            }
            Self::SampledArrayElementOutOfRange {
                binding,
                element,
                count,
            }
            | Self::StorageArrayElementOutOfRange {
                binding,
                element,
                count,
            } => vec![
                ("binding", binding.to_string()),
                ("element", element.to_string()),
                ("count", count.to_string()),
            ],
            Self::SampledZeroGeometry {
                binding,
                width,
                height,
            }
            | Self::StorageZeroGeometry {
                binding,
                width,
                height,
            } => vec![
                ("binding", binding.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
            ],
            Self::SampledBytesLength {
                binding,
                actual,
                expected,
            }
            | Self::StorageBytesLength {
                binding,
                actual,
                expected,
            } => vec![
                ("binding", binding.to_string()),
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::InvalidSamplerLod {
                binding,
                lod_min_bits,
                lod_max_bits,
            } => vec![
                ("binding", binding.to_string()),
                ("lod_min", f32::from_bits(*lod_min_bits).to_string()),
                ("lod_max", f32::from_bits(*lod_max_bits).to_string()),
            ],
            Self::ProgramUnavailable { id } => vec![("id", id.to_string())],
            Self::MissingProgram | Self::EmptyEntry | Self::EntryInteriorNul => Vec::new(),
        }
    }
}

reims_vgpu_observe::decline_display!(ComputeValidationDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<ComputeValidationDecline> {
        vec![
            ComputeValidationDecline::MissingProgram,
            ComputeValidationDecline::ProgramUnavailable { id: 1 },
            ComputeValidationDecline::EmptyEntry,
            ComputeValidationDecline::EntryInteriorNul,
            ComputeValidationDecline::ZeroGrid { grid: [1, 0, 1] },
            ComputeValidationDecline::DuplicateStorageBufferBinding { binding: 0 },
            ComputeValidationDecline::EmptyStorageBuffer { binding: 0 },
            ComputeValidationDecline::DuplicateSampledImageBinding { binding: 32 },
            ComputeValidationDecline::SampledArrayElementOutOfRange {
                binding: 32,
                element: 2,
                count: 2,
            },
            ComputeValidationDecline::SampledZeroGeometry {
                binding: 32,
                width: 0,
                height: 1,
            },
            ComputeValidationDecline::SampledBytesLength {
                binding: 32,
                actual: 3,
                expected: 4,
            },
            ComputeValidationDecline::InvalidSamplerLod {
                binding: 64,
                lod_min_bits: 2.0f32.to_bits(),
                lod_max_bits: 1.0f32.to_bits(),
            },
            ComputeValidationDecline::DuplicateSamplerBinding { binding: 64 },
            ComputeValidationDecline::DuplicateStorageImageBinding { binding: 34 },
            ComputeValidationDecline::StorageArrayElementOutOfRange {
                binding: 34,
                element: 2,
                count: 2,
            },
            ComputeValidationDecline::StorageZeroGeometry {
                binding: 34,
                width: 1,
                height: 0,
            },
            ComputeValidationDecline::StorageBytesLength {
                binding: 34,
                actual: 3,
                expected: 4,
            },
        ]
    }

    #[test]
    fn every_compute_validation_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_compute_validate_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        // Down from 18: the four 1D/array-layer checks went out with the
        // non-2D image shape. A compute texture binding is one flat plane
        // window or one linear GVA level, so there is no slice or depth axis
        // for a request to get wrong.
        assert_eq!(before, 17, "the compute validator's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate compute-validation slug");
    }

    #[test]
    fn compute_validation_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line =
                reims_vgpu_observe::Emit::decline("compute_validation_test", &decline).render();
            assert!(line.starts_with(&format!(
                "compute_validation_test reason={}",
                decline.slug()
            )));
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
