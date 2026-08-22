//! Typed validation failures for a Vulkan draw request.
//!
//! These checks run before context creation or GPU work. They used to be the
//! largest single `DrawError::Invalid(String)` cluster: 41 constructors in
//! `exec`, with buffer and descriptor roles collapsed into identical prose.
//! The typed vocabulary makes each request invariant enumerable and preserves
//! the geometry/binding/range values needed to reproduce it.

use ash::vk;

use reims_vgpu_observe::Decline;

/// A specific malformed or internally inconsistent draw request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawValidationDecline {
    VertexGuestRunsRowStride {
        location: u32,
        row_length_texels: u32,
    },
    StorageGuestRunsRowStride {
        binding: u32,
        row_length_texels: u32,
    },
    IndexGuestRunsRowStride {
        row_length_texels: u32,
    },
    VertexGuestRunsCoverage {
        location: u32,
        covered: u64,
        declared: u64,
    },
    StorageGuestRunsCoverage {
        binding: u32,
        covered: u64,
        declared: u64,
    },
    IndexGuestRunsCoverage {
        covered: u64,
        declared: u64,
    },
    ZeroTargetGeometry {
        width: u32,
        height: u32,
    },
    RenderTargetExtentExceedsAttachment {
        axis: &'static str,
        requested: u32,
        limit: u32,
    },
    MissingVertexProgram,
    MissingFragmentProgram,
    VertexProgramUnavailable {
        id: u64,
    },
    FragmentProgramUnavailable {
        id: u64,
    },
    NonFiniteViewport,
    NonPositiveViewport {
        width_bits: u32,
        height_bits: u32,
    },
    NonFiniteBlendConstants,
    TargetSeedLength {
        actual: usize,
        expected: usize,
    },
    TargetGuestSeedFormat {
        source: vk::Format,
        target: vk::Format,
    },
    TargetGuestSeedRowStride {
        stride: usize,
        tight_row: usize,
    },
    TargetGuestSeedLength {
        actual: u64,
        expected: usize,
    },
    TargetGuestSeedCoverage {
        covered: u64,
        declared: u64,
    },
    SeedConflictsGuestSeed,
    SeedMissingTargetIdentity,
    SeedConflictsCpuSeed,
    SeedConflictsLoadFromTarget,
    SeedEqualsTarget,
    SeedAlsoSampled,
    IndexBytesShort {
        actual: usize,
        expected: usize,
    },
    DuplicateVertexLocation {
        location: u32,
    },
    DuplicateVertexBinding {
        binding: u32,
    },
    ZeroVertexStepRate {
        location: u32,
    },
    VertexStrideTooSmall {
        location: u32,
        stride: u32,
        format_size: u32,
    },
    VertexOffsetOverflow {
        location: u32,
    },
    VertexElementExceedsStride {
        location: u32,
    },
    VertexRangeOverflow {
        location: u32,
    },
    InstanceRangeOverflow {
        location: u32,
    },
    VertexByteRangeOverflow {
        location: u32,
    },
    VertexDataShort {
        location: u32,
        actual: usize,
        expected: usize,
    },
    ConstantStepGuestRuns {
        location: u32,
    },
    DuplicateStorageDescriptorBinding {
        binding: u32,
    },
    DuplicateSampledDescriptorBinding {
        binding: u32,
    },
    SampledArrayElementOutOfRange {
        binding: u32,
        element: u32,
        count: u32,
    },
    DuplicateSamplerDescriptorBinding {
        binding: u32,
    },
    SampledZeroGeometry {
        binding: u32,
        width: u32,
        height: u32,
        layers: u32,
    },
    SampledShapeConflict {
        binding: u32,
        arrayed: bool,
        volume: bool,
        cube: bool,
    },
    SampledCubeGeometry {
        binding: u32,
        width: u32,
        height: u32,
        layers: u32,
    },
    SampledNonArrayLayers {
        binding: u32,
        layers: u32,
    },
    SampledBytesLength {
        binding: u32,
        actual: usize,
        expected: usize,
    },
    /// The tightly-packed length this geometry implies does not fit a `usize`,
    /// so no buffer can be that long and no comparison against one would mean
    /// anything.
    ///
    /// Reached only from a decoded geometry, and it must be a refusal rather
    /// than a clamp: the length is what the *next* check compares a buffer
    /// against, and a wrapped one would let a short buffer match. It used to be
    /// neither — the product was taken unchecked, which panics in a debug build
    /// from inside the function whose whole job is to survive a malformed
    /// request.
    UnrepresentableImageBytes {
        width: u32,
        height: u32,
        layers: u32,
        bytes_per_texel: u32,
    },
    ResidentSampleGeometry {
        binding: u32,
        resident_width: u32,
        resident_height: u32,
        resource_width: u32,
        resource_height: u32,
    },
    GuestSampleRowStride {
        binding: u32,
        stride: usize,
        tight_row: usize,
    },
    GuestSampleLayoutMismatch {
        binding: u32,
        layout: reims_vgpu_memory::GuestImageLayout,
        width: u32,
        height: u32,
        layers: u32,
        arrayed: bool,
        volume: bool,
        one_dim: bool,
        multisampled: bool,
    },
    GuestSampleViewRangeInvalid {
        binding: u32,
        view: reims_vgpu_memory::GuestImageViewRange,
        mip_levels: usize,
    },
    GuestSampleAllocationInvalid {
        binding: u32,
        mip_levels: usize,
        bytes_per_texel: u64,
    },
    GuestSampleLayoutInvalid {
        binding: u32,
        layout: reims_vgpu_memory::GuestImageLayout,
        row_pitch: u64,
        bytes_per_texel: u64,
    },
    GuestSampleLength {
        binding: u32,
        actual: u64,
        expected: u64,
    },
    GuestSampleCoverageOverflow {
        binding: u32,
        runs: usize,
    },
    GuestSampleCoverage {
        binding: u32,
        covered: u64,
        declared: u64,
        runs: usize,
    },
    InvalidSamplerLod {
        binding: u32,
        lod_min_bits: u32,
        lod_max_bits: u32,
    },
}

impl Decline for DrawValidationDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::VertexGuestRunsRowStride { .. } => {
                "vk_draw_validate_vertex_guest_runs_row_stride"
            }
            Self::StorageGuestRunsRowStride { .. } => {
                "vk_draw_validate_storage_guest_runs_row_stride"
            }
            Self::VertexGuestRunsCoverage { .. } => "vk_draw_validate_vertex_guest_runs_coverage",
            Self::StorageGuestRunsCoverage { .. } => "vk_draw_validate_storage_guest_runs_coverage",
            Self::ZeroTargetGeometry { .. } => "vk_draw_validate_zero_target_geometry",
            Self::RenderTargetExtentExceedsAttachment { .. } => {
                "vk_draw_validate_render_target_extent_exceeds_attachment"
            }
            Self::MissingVertexProgram => "vk_draw_validate_missing_vertex_program",
            Self::MissingFragmentProgram => "vk_draw_validate_missing_fragment_program",
            Self::VertexProgramUnavailable { .. } => "vk_draw_validate_vertex_program_unavailable",
            Self::FragmentProgramUnavailable { .. } => {
                "vk_draw_validate_fragment_program_unavailable"
            }
            Self::NonFiniteViewport => "vk_draw_validate_non_finite_viewport",
            Self::NonPositiveViewport { .. } => "vk_draw_validate_non_positive_viewport",
            Self::NonFiniteBlendConstants => "vk_draw_validate_non_finite_blend_constants",
            Self::TargetSeedLength { .. } => "vk_draw_validate_target_seed_length",
            Self::TargetGuestSeedFormat { .. } => "vk_draw_validate_target_guest_seed_format",
            Self::TargetGuestSeedRowStride { .. } => {
                "vk_draw_validate_target_guest_seed_row_stride"
            }
            Self::TargetGuestSeedLength { .. } => "vk_draw_validate_target_guest_seed_length",
            Self::TargetGuestSeedCoverage { .. } => "vk_draw_validate_target_guest_seed_coverage",
            Self::SeedConflictsGuestSeed => "vk_draw_validate_seed_conflicts_guest_seed",
            Self::SeedMissingTargetIdentity => "vk_draw_validate_seed_missing_target_identity",
            Self::SeedConflictsCpuSeed => "vk_draw_validate_seed_conflicts_cpu_seed",
            Self::SeedConflictsLoadFromTarget => "vk_draw_validate_seed_conflicts_load_from_target",
            Self::SeedEqualsTarget => "vk_draw_validate_seed_equals_target",
            Self::SeedAlsoSampled => "vk_draw_validate_seed_also_sampled",
            Self::IndexBytesShort { .. } => "vk_draw_validate_index_bytes_short",
            Self::IndexGuestRunsRowStride { .. } => "vk_draw_validate_index_guest_runs_row_stride",
            Self::IndexGuestRunsCoverage { .. } => "vk_draw_validate_index_guest_runs_coverage",
            Self::DuplicateVertexLocation { .. } => "vk_draw_validate_duplicate_vertex_location",
            Self::DuplicateVertexBinding { .. } => "vk_draw_validate_duplicate_vertex_binding",
            Self::ZeroVertexStepRate { .. } => "vk_draw_validate_zero_vertex_step_rate",
            Self::VertexStrideTooSmall { .. } => "vk_draw_validate_vertex_stride_too_small",
            Self::VertexOffsetOverflow { .. } => "vk_draw_validate_vertex_offset_overflow",
            Self::VertexElementExceedsStride { .. } => {
                "vk_draw_validate_vertex_element_exceeds_stride"
            }
            Self::VertexRangeOverflow { .. } => "vk_draw_validate_vertex_range_overflow",
            Self::InstanceRangeOverflow { .. } => "vk_draw_validate_instance_range_overflow",
            Self::VertexByteRangeOverflow { .. } => "vk_draw_validate_vertex_byte_range_overflow",
            Self::VertexDataShort { .. } => "vk_draw_validate_vertex_data_short",
            Self::ConstantStepGuestRuns { .. } => "vk_draw_validate_constant_step_guest_runs",
            Self::DuplicateStorageDescriptorBinding { .. } => {
                "vk_draw_validate_duplicate_storage_descriptor_binding"
            }
            Self::DuplicateSampledDescriptorBinding { .. } => {
                "vk_draw_validate_duplicate_sampled_descriptor_binding"
            }
            Self::SampledArrayElementOutOfRange { .. } => {
                "vk_draw_validate_sampled_array_element_out_of_range"
            }
            Self::DuplicateSamplerDescriptorBinding { .. } => {
                "vk_draw_validate_duplicate_sampler_descriptor_binding"
            }
            Self::SampledZeroGeometry { .. } => "vk_draw_validate_sampled_zero_geometry",
            Self::SampledShapeConflict { .. } => "vk_draw_validate_sampled_shape_conflict",
            Self::SampledCubeGeometry { .. } => "vk_draw_validate_sampled_cube_geometry",
            Self::SampledNonArrayLayers { .. } => "vk_draw_validate_sampled_nonarray_layers",
            Self::SampledBytesLength { .. } => "vk_draw_validate_sampled_bytes_length",
            Self::UnrepresentableImageBytes { .. } => {
                "vk_draw_validate_unrepresentable_image_bytes"
            }
            Self::ResidentSampleGeometry { .. } => "vk_draw_validate_resident_sample_geometry",
            Self::GuestSampleRowStride { .. } => "vk_draw_validate_guest_sample_row_stride",
            Self::GuestSampleLayoutMismatch { .. } => {
                "vk_draw_validate_guest_sample_layout_mismatch"
            }
            Self::GuestSampleViewRangeInvalid { .. } => {
                "vk_draw_validate_guest_sample_view_range_invalid"
            }
            Self::GuestSampleAllocationInvalid { .. } => {
                "vk_draw_validate_guest_sample_allocation_invalid"
            }
            Self::GuestSampleLayoutInvalid { .. } => "vk_draw_validate_guest_sample_layout_invalid",
            Self::GuestSampleLength { .. } => "vk_draw_validate_guest_sample_length",
            Self::GuestSampleCoverageOverflow { .. } => {
                "vk_draw_validate_guest_sample_coverage_overflow"
            }
            Self::GuestSampleCoverage { .. } => "vk_draw_validate_guest_sample_coverage",
            Self::InvalidSamplerLod { .. } => "vk_draw_validate_invalid_sampler_lod",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::VertexGuestRunsRowStride {
                location,
                row_length_texels,
            } => vec![
                ("location", location.to_string()),
                ("row_length_texels", row_length_texels.to_string()),
            ],
            Self::StorageGuestRunsRowStride {
                binding,
                row_length_texels,
            } => vec![
                ("binding", binding.to_string()),
                ("row_length_texels", row_length_texels.to_string()),
            ],
            Self::VertexGuestRunsCoverage {
                location,
                covered,
                declared,
            } => vec![
                ("location", location.to_string()),
                ("covered", covered.to_string()),
                ("declared", declared.to_string()),
            ],
            Self::StorageGuestRunsCoverage {
                binding,
                covered,
                declared,
            } => vec![
                ("binding", binding.to_string()),
                ("covered", covered.to_string()),
                ("declared", declared.to_string()),
            ],
            Self::IndexGuestRunsRowStride { row_length_texels } => {
                vec![("row_length_texels", row_length_texels.to_string())]
            }
            Self::IndexGuestRunsCoverage { covered, declared } => vec![
                ("covered", covered.to_string()),
                ("declared", declared.to_string()),
            ],
            Self::ZeroTargetGeometry { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
            Self::RenderTargetExtentExceedsAttachment {
                axis,
                requested,
                limit,
            } => vec![
                ("axis", (*axis).into()),
                ("requested", requested.to_string()),
                ("limit", limit.to_string()),
            ],
            Self::NonPositiveViewport {
                width_bits,
                height_bits,
            } => vec![
                ("width", f32::from_bits(*width_bits).to_string()),
                ("height", f32::from_bits(*height_bits).to_string()),
            ],
            Self::TargetSeedLength { actual, expected }
            | Self::IndexBytesShort { actual, expected } => vec![
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::TargetGuestSeedFormat { source, target } => vec![
                ("source", format!("{source:?}")),
                ("target", format!("{target:?}")),
            ],
            Self::TargetGuestSeedRowStride { stride, tight_row } => vec![
                ("stride", stride.to_string()),
                ("tight_row", tight_row.to_string()),
            ],
            Self::TargetGuestSeedLength { actual, expected } => vec![
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::TargetGuestSeedCoverage { covered, declared } => vec![
                ("covered", covered.to_string()),
                ("declared", declared.to_string()),
            ],
            Self::DuplicateVertexLocation { location }
            | Self::ZeroVertexStepRate { location }
            | Self::VertexOffsetOverflow { location }
            | Self::VertexElementExceedsStride { location }
            | Self::VertexRangeOverflow { location }
            | Self::InstanceRangeOverflow { location }
            | Self::VertexByteRangeOverflow { location }
            | Self::ConstantStepGuestRuns { location } => {
                vec![("location", location.to_string())]
            }
            Self::DuplicateVertexBinding { binding }
            | Self::DuplicateStorageDescriptorBinding { binding }
            | Self::DuplicateSampledDescriptorBinding { binding }
            | Self::DuplicateSamplerDescriptorBinding { binding } => {
                vec![("binding", binding.to_string())]
            }
            Self::SampledArrayElementOutOfRange {
                binding,
                element,
                count,
            } => vec![
                ("binding", binding.to_string()),
                ("element", element.to_string()),
                ("count", count.to_string()),
            ],
            Self::VertexStrideTooSmall {
                location,
                stride,
                format_size,
            } => vec![
                ("location", location.to_string()),
                ("stride", stride.to_string()),
                ("format_size", format_size.to_string()),
            ],
            Self::VertexDataShort {
                location,
                actual,
                expected,
            } => vec![
                ("location", location.to_string()),
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::SampledZeroGeometry {
                binding,
                width,
                height,
                layers,
            }
            | Self::SampledCubeGeometry {
                binding,
                width,
                height,
                layers,
            } => vec![
                ("binding", binding.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("layers", layers.to_string()),
            ],
            Self::SampledShapeConflict {
                binding,
                arrayed,
                volume,
                cube,
            } => vec![
                ("binding", binding.to_string()),
                ("arrayed", arrayed.to_string()),
                ("volume", volume.to_string()),
                ("cube", cube.to_string()),
            ],
            Self::SampledNonArrayLayers { binding, layers } => vec![
                ("binding", binding.to_string()),
                ("layers", layers.to_string()),
            ],
            Self::SampledBytesLength {
                binding,
                actual,
                expected,
            } => vec![
                ("binding", binding.to_string()),
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::UnrepresentableImageBytes {
                width,
                height,
                layers,
                bytes_per_texel,
            } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("layers", layers.to_string()),
                ("bytes_per_texel", bytes_per_texel.to_string()),
            ],
            Self::ResidentSampleGeometry {
                binding,
                resident_width,
                resident_height,
                resource_width,
                resource_height,
            } => vec![
                ("binding", binding.to_string()),
                ("resident_width", resident_width.to_string()),
                ("resident_height", resident_height.to_string()),
                ("resource_width", resource_width.to_string()),
                ("resource_height", resource_height.to_string()),
            ],
            Self::GuestSampleRowStride {
                binding,
                stride,
                tight_row,
            } => vec![
                ("binding", binding.to_string()),
                ("stride", stride.to_string()),
                ("tight_row", tight_row.to_string()),
            ],
            Self::GuestSampleLayoutMismatch {
                binding,
                layout,
                width,
                height,
                layers,
                arrayed,
                volume,
                one_dim,
                multisampled,
            } => vec![
                ("binding", binding.to_string()),
                ("layout", format!("{layout:?}")),
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("layers", layers.to_string()),
                ("arrayed", arrayed.to_string()),
                ("volume", volume.to_string()),
                ("one_dim", one_dim.to_string()),
                ("multisampled", multisampled.to_string()),
            ],
            Self::GuestSampleViewRangeInvalid {
                binding,
                view,
                mip_levels,
            } => vec![
                ("binding", binding.to_string()),
                ("view", format!("{view:?}")),
                ("mip_levels", mip_levels.to_string()),
            ],
            Self::GuestSampleAllocationInvalid {
                binding,
                mip_levels,
                bytes_per_texel,
            } => vec![
                ("binding", binding.to_string()),
                ("mip_levels", mip_levels.to_string()),
                ("bytes_per_texel", bytes_per_texel.to_string()),
            ],
            Self::GuestSampleLayoutInvalid {
                binding,
                layout,
                row_pitch,
                bytes_per_texel,
            } => vec![
                ("binding", binding.to_string()),
                ("layout", format!("{layout:?}")),
                ("row_pitch", row_pitch.to_string()),
                ("bytes_per_texel", bytes_per_texel.to_string()),
            ],
            Self::GuestSampleLength {
                binding,
                actual,
                expected,
            } => vec![
                ("binding", binding.to_string()),
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::GuestSampleCoverageOverflow { binding, runs } => {
                vec![("binding", binding.to_string()), ("runs", runs.to_string())]
            }
            Self::GuestSampleCoverage {
                binding,
                covered,
                declared,
                runs,
            } => vec![
                ("binding", binding.to_string()),
                ("covered", covered.to_string()),
                ("declared", declared.to_string()),
                ("runs", runs.to_string()),
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
            Self::VertexProgramUnavailable { id } | Self::FragmentProgramUnavailable { id } => {
                vec![("id", id.to_string())]
            }
            Self::MissingVertexProgram
            | Self::MissingFragmentProgram
            | Self::NonFiniteViewport
            | Self::NonFiniteBlendConstants
            | Self::SeedConflictsGuestSeed
            | Self::SeedMissingTargetIdentity
            | Self::SeedConflictsCpuSeed
            | Self::SeedConflictsLoadFromTarget
            | Self::SeedEqualsTarget
            | Self::SeedAlsoSampled => Vec::new(),
        }
    }
}

reims_vgpu_observe::decline_display!(DrawValidationDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<DrawValidationDecline> {
        vec![
            DrawValidationDecline::VertexGuestRunsRowStride {
                location: 0,
                row_length_texels: 16,
            },
            DrawValidationDecline::StorageGuestRunsRowStride {
                binding: 1,
                row_length_texels: 16,
            },
            DrawValidationDecline::IndexGuestRunsRowStride {
                row_length_texels: 16,
            },
            DrawValidationDecline::VertexGuestRunsCoverage {
                location: 0,
                covered: 3,
                declared: 4,
            },
            DrawValidationDecline::StorageGuestRunsCoverage {
                binding: 1,
                covered: 3,
                declared: 4,
            },
            DrawValidationDecline::IndexGuestRunsCoverage {
                covered: 3,
                declared: 4,
            },
            DrawValidationDecline::ZeroTargetGeometry {
                width: 0,
                height: 8,
            },
            DrawValidationDecline::MissingVertexProgram,
            DrawValidationDecline::MissingFragmentProgram,
            DrawValidationDecline::VertexProgramUnavailable { id: 1 },
            DrawValidationDecline::FragmentProgramUnavailable { id: 2 },
            DrawValidationDecline::NonFiniteViewport,
            DrawValidationDecline::NonPositiveViewport {
                width_bits: 0.0f32.to_bits(),
                height_bits: 1.0f32.to_bits(),
            },
            DrawValidationDecline::NonFiniteBlendConstants,
            DrawValidationDecline::TargetSeedLength {
                actual: 3,
                expected: 4,
            },
            DrawValidationDecline::TargetGuestSeedFormat {
                source: vk::Format::R8_UNORM,
                target: vk::Format::R8G8_UNORM,
            },
            DrawValidationDecline::TargetGuestSeedRowStride {
                stride: 4,
                tight_row: 8,
            },
            DrawValidationDecline::TargetGuestSeedLength {
                actual: 3,
                expected: 4,
            },
            DrawValidationDecline::TargetGuestSeedCoverage {
                covered: 3,
                declared: 4,
            },
            DrawValidationDecline::SeedConflictsGuestSeed,
            DrawValidationDecline::SeedMissingTargetIdentity,
            DrawValidationDecline::SeedConflictsCpuSeed,
            DrawValidationDecline::SeedConflictsLoadFromTarget,
            DrawValidationDecline::SeedEqualsTarget,
            DrawValidationDecline::SeedAlsoSampled,
            DrawValidationDecline::IndexBytesShort {
                actual: 3,
                expected: 4,
            },
            DrawValidationDecline::DuplicateVertexLocation { location: 0 },
            DrawValidationDecline::DuplicateVertexBinding { binding: 0 },
            DrawValidationDecline::ZeroVertexStepRate { location: 0 },
            DrawValidationDecline::VertexStrideTooSmall {
                location: 0,
                stride: 4,
                format_size: 8,
            },
            DrawValidationDecline::VertexOffsetOverflow { location: 0 },
            DrawValidationDecline::VertexElementExceedsStride { location: 0 },
            DrawValidationDecline::VertexRangeOverflow { location: 0 },
            DrawValidationDecline::InstanceRangeOverflow { location: 0 },
            DrawValidationDecline::VertexByteRangeOverflow { location: 0 },
            DrawValidationDecline::VertexDataShort {
                location: 0,
                actual: 3,
                expected: 4,
            },
            DrawValidationDecline::ConstantStepGuestRuns { location: 0 },
            DrawValidationDecline::DuplicateStorageDescriptorBinding { binding: 0 },
            DrawValidationDecline::DuplicateSampledDescriptorBinding { binding: 0 },
            DrawValidationDecline::SampledArrayElementOutOfRange {
                binding: 32,
                element: 2,
                count: 2,
            },
            DrawValidationDecline::DuplicateSamplerDescriptorBinding { binding: 0 },
            DrawValidationDecline::SampledZeroGeometry {
                binding: 32,
                width: 0,
                height: 8,
                layers: 1,
            },
            DrawValidationDecline::SampledShapeConflict {
                binding: 32,
                arrayed: true,
                volume: true,
                cube: false,
            },
            DrawValidationDecline::SampledCubeGeometry {
                binding: 32,
                width: 8,
                height: 4,
                layers: 5,
            },
            DrawValidationDecline::SampledNonArrayLayers {
                binding: 32,
                layers: 2,
            },
            DrawValidationDecline::SampledBytesLength {
                binding: 32,
                actual: 3,
                expected: 4,
            },
            DrawValidationDecline::UnrepresentableImageBytes {
                width: 32,
                height: 32,
                layers: 32,
                bytes_per_texel: 32,
            },
            DrawValidationDecline::ResidentSampleGeometry {
                binding: 32,
                resident_width: 8,
                resident_height: 8,
                resource_width: 4,
                resource_height: 4,
            },
            DrawValidationDecline::GuestSampleRowStride {
                binding: 32,
                stride: 4,
                tight_row: 8,
            },
            DrawValidationDecline::GuestSampleLayoutMismatch {
                binding: 32,
                layout: reims_vgpu_memory::GuestImageLayout::D3 {
                    width: 1,
                    height: 1,
                    depth: 1,
                    depth_pitch: 4,
                },
                width: 1,
                height: 1,
                layers: 1,
                arrayed: false,
                volume: false,
                one_dim: false,
                multisampled: false,
            },
            DrawValidationDecline::GuestSampleLayoutInvalid {
                binding: 32,
                layout: reims_vgpu_memory::GuestImageLayout::D2 {
                    width: 8,
                    height: 8,
                },
                row_pitch: 4,
                bytes_per_texel: 4,
            },
            DrawValidationDecline::GuestSampleAllocationInvalid {
                binding: 32,
                mip_levels: 2,
                bytes_per_texel: 4,
            },
            DrawValidationDecline::GuestSampleLength {
                binding: 32,
                actual: 3,
                expected: 4,
            },
            DrawValidationDecline::GuestSampleCoverageOverflow {
                binding: 32,
                runs: 2,
            },
            DrawValidationDecline::GuestSampleCoverage {
                binding: 32,
                covered: 3,
                declared: 4,
                runs: 1,
            },
            DrawValidationDecline::InvalidSamplerLod {
                binding: 64,
                lod_min_bits: 2.0f32.to_bits(),
                lod_max_bits: 1.0f32.to_bits(),
            },
        ]
    }

    #[test]
    fn every_draw_validation_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_draw_validate_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        // Typed guest-image checks preserve allocation, view, layout, pitch,
        // and run coverage as distinct failures.
        assert_eq!(before, 56, "the validator's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate draw-validation slug");
    }

    #[test]
    fn draw_validation_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = reims_vgpu_observe::Emit::decline("draw_validation_test", &decline).render();
            assert!(line.starts_with(&format!("draw_validation_test reason={}", decline.slug())));
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
