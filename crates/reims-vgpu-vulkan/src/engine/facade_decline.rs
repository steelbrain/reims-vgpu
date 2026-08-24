//! Typed failures at the Vulkan engine façade and host-window presenter seam.
//!
//! These checks are neither malformed draw/compute requests nor failed Vulkan
//! calls. They reject an engine entry point because the façade's tracked state
//! disappeared or disagreed with the caller — the named resident is absent, is
//! at the wrong generation, or is not yet content-ready.

use super::compute_execution::residency_fields;
use super::draw_execution::identity_fields;
use super::types::TargetIdentity;
use reims_vgpu_core::ComputeStorageResidencyKey;
use reims_vgpu_observe::Decline;
use reims_vgpu_protocol::SubmissionIdentity;

/// A specific engine façade or host-window presenter state failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineFacadeDecline {
    ExecutorServiceUnavailable {
        service: &'static str,
    },
    ExecutorCompletionKindMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    ExecutorCompletionCountMismatch {
        expected: usize,
        actual: usize,
    },
    ExecutorCompletionIdentityMismatch {
        expected: SubmissionIdentity,
        actual: SubmissionIdentity,
    },
    EncoderSubmissionOverlap {
        active: SubmissionIdentity,
        incoming: SubmissionIdentity,
    },
    EncoderSubmissionCloseMismatch {
        active: SubmissionIdentity,
        closing: SubmissionIdentity,
    },
    WindowPresenterNotAttached,
    StorageReadResidentAbsent {
        identity: ComputeStorageResidencyKey,
    },
    StorageReadGenerationMismatch {
        identity: ComputeStorageResidencyKey,
        actual_generation: u32,
        expected_generation: u32,
    },
    WindowSourceDisappearedBeforePin {
        identity: TargetIdentity,
    },
    ResidentCopySameIdentity {
        identity: TargetIdentity,
    },
    ResidentCopyGeometryMismatch {
        source_width: u32,
        source_height: u32,
        destination_width: u32,
        destination_height: u32,
    },
    ResidentCopyFormatMismatch {
        source: ash::vk::Format,
        destination: ash::vk::Format,
    },
    ResidentCopyDestinationUnavailable {
        identity: TargetIdentity,
    },
    ResidentCopyPinRefused {
        identity: TargetIdentity,
    },
}

impl Decline for EngineFacadeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ExecutorServiceUnavailable { .. } => "vk_engine_executor_service_unavailable",
            Self::ExecutorCompletionKindMismatch { .. } => {
                "vk_engine_executor_completion_kind_mismatch"
            }
            Self::ExecutorCompletionCountMismatch { .. } => {
                "vk_engine_executor_completion_count_mismatch"
            }
            Self::ExecutorCompletionIdentityMismatch { .. } => {
                "vk_engine_executor_completion_identity_mismatch"
            }
            Self::EncoderSubmissionOverlap { .. } => "vk_engine_encoder_submission_overlap",
            Self::EncoderSubmissionCloseMismatch { .. } => {
                "vk_engine_encoder_submission_close_mismatch"
            }
            Self::WindowPresenterNotAttached => "vk_engine_window_presenter_not_attached",
            Self::StorageReadResidentAbsent { .. } => "vk_engine_storage_read_resident_absent",
            Self::StorageReadGenerationMismatch { .. } => {
                "vk_engine_storage_read_generation_mismatch"
            }
            Self::WindowSourceDisappearedBeforePin { .. } => {
                "vk_engine_window_source_disappeared_before_pin"
            }
            Self::ResidentCopySameIdentity { .. } => "vk_engine_resident_copy_same_identity",
            Self::ResidentCopyGeometryMismatch { .. } => {
                "vk_engine_resident_copy_geometry_mismatch"
            }
            Self::ResidentCopyFormatMismatch { .. } => "vk_engine_resident_copy_format_mismatch",
            Self::ResidentCopyDestinationUnavailable { .. } => {
                "vk_engine_resident_copy_destination_unavailable"
            }
            Self::ResidentCopyPinRefused { .. } => "vk_engine_resident_copy_pin_refused",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ExecutorServiceUnavailable { service } => {
                vec![("service", (*service).to_string())]
            }
            Self::ExecutorCompletionKindMismatch { expected, actual } => vec![
                ("expected", (*expected).to_string()),
                ("actual", (*actual).to_string()),
            ],
            Self::ExecutorCompletionCountMismatch { expected, actual } => vec![
                ("expected", expected.to_string()),
                ("actual", actual.to_string()),
            ],
            Self::ExecutorCompletionIdentityMismatch { expected, actual } => vec![
                ("expected_submission", expected.id.get().to_string()),
                ("expected_task", expected.task.get().to_string()),
                ("actual_submission", actual.id.get().to_string()),
                ("actual_task", actual.task.get().to_string()),
            ],
            Self::EncoderSubmissionOverlap { active, incoming } => vec![
                ("active_submission", active.id.get().to_string()),
                ("active_task", active.task.get().to_string()),
                ("incoming_submission", incoming.id.get().to_string()),
                ("incoming_task", incoming.task.get().to_string()),
            ],
            Self::EncoderSubmissionCloseMismatch { active, closing } => vec![
                ("active_submission", active.id.get().to_string()),
                ("active_task", active.task.get().to_string()),
                ("closing_submission", closing.id.get().to_string()),
                ("closing_task", closing.task.get().to_string()),
            ],
            Self::WindowPresenterNotAttached => Vec::new(),
            Self::StorageReadResidentAbsent { identity } => residency_fields(identity),
            Self::StorageReadGenerationMismatch {
                identity,
                actual_generation,
                expected_generation,
            } => {
                let mut fields = residency_fields(identity);
                fields.extend([
                    ("actual_generation", actual_generation.to_string()),
                    ("expected_generation", expected_generation.to_string()),
                ]);
                fields
            }
            Self::WindowSourceDisappearedBeforePin { identity } => identity_fields(identity),
            Self::ResidentCopySameIdentity { identity }
            | Self::ResidentCopyDestinationUnavailable { identity }
            | Self::ResidentCopyPinRefused { identity } => identity_fields(identity),
            Self::ResidentCopyGeometryMismatch {
                source_width,
                source_height,
                destination_width,
                destination_height,
            } => vec![
                ("source_width", source_width.to_string()),
                ("source_height", source_height.to_string()),
                ("destination_width", destination_width.to_string()),
                ("destination_height", destination_height.to_string()),
            ],
            Self::ResidentCopyFormatMismatch {
                source,
                destination,
            } => vec![
                ("source_format", format!("{source:?}")),
                ("destination_format", format!("{destination:?}")),
            ],
        }
    }
}

reims_vgpu_observe::decline_display!(EngineFacadeDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn residency() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey::surface(7, 8, 0x9000, 256, 4096, 64, 32, 80)
    }

    fn identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 7,
            width: 64,
            height: 32,
            generation: 9,
            format: reims_vgpu_protocol::TexelLayout::Bgra8,
        }
    }

    fn all() -> Vec<EngineFacadeDecline> {
        vec![
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "target_readback",
            },
            EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: "draw",
                actual: "compute",
            },
            EngineFacadeDecline::ExecutorCompletionCountMismatch {
                expected: 1,
                actual: 2,
            },
            EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected: SubmissionIdentity {
                    id: reims_vgpu_protocol::SubmissionId::new(1),
                    task: reims_vgpu_protocol::TaskId::new(2),
                },
                actual: SubmissionIdentity {
                    id: reims_vgpu_protocol::SubmissionId::new(3),
                    task: reims_vgpu_protocol::TaskId::new(4),
                },
            },
            EngineFacadeDecline::EncoderSubmissionOverlap {
                active: SubmissionIdentity {
                    id: reims_vgpu_protocol::SubmissionId::new(5),
                    task: reims_vgpu_protocol::TaskId::new(6),
                },
                incoming: SubmissionIdentity {
                    id: reims_vgpu_protocol::SubmissionId::new(7),
                    task: reims_vgpu_protocol::TaskId::new(8),
                },
            },
            EngineFacadeDecline::EncoderSubmissionCloseMismatch {
                active: SubmissionIdentity {
                    id: reims_vgpu_protocol::SubmissionId::new(9),
                    task: reims_vgpu_protocol::TaskId::new(10),
                },
                closing: SubmissionIdentity {
                    id: reims_vgpu_protocol::SubmissionId::new(11),
                    task: reims_vgpu_protocol::TaskId::new(12),
                },
            },
            EngineFacadeDecline::WindowPresenterNotAttached,
            EngineFacadeDecline::StorageReadResidentAbsent {
                identity: residency(),
            },
            EngineFacadeDecline::StorageReadGenerationMismatch {
                identity: residency(),
                actual_generation: 8,
                expected_generation: 9,
            },
            EngineFacadeDecline::WindowSourceDisappearedBeforePin {
                identity: identity(),
            },
            EngineFacadeDecline::ResidentCopySameIdentity {
                identity: identity(),
            },
            EngineFacadeDecline::ResidentCopyGeometryMismatch {
                source_width: 64,
                source_height: 32,
                destination_width: 32,
                destination_height: 16,
            },
            EngineFacadeDecline::ResidentCopyFormatMismatch {
                source: ash::vk::Format::R8G8B8A8_UNORM,
                destination: ash::vk::Format::B8G8R8A8_UNORM,
            },
            EngineFacadeDecline::ResidentCopyDestinationUnavailable {
                identity: identity(),
            },
            EngineFacadeDecline::ResidentCopyPinRefused {
                identity: identity(),
            },
        ]
    }

    #[test]
    fn every_engine_facade_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_engine_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 15, "the engine façade reason census moved");
        assert_eq!(before, slugs.len(), "duplicate engine façade slug");
    }

    #[test]
    fn engine_facade_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = reims_vgpu_observe::Emit::decline("engine_facade_test", &decline).render();
            assert!(line.starts_with(&format!("engine_facade_test reason={}", decline.slug())));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }
    }

    /// The identity fields are what a reader needs to find the resident a
    /// window-source pin missed, so they must all reach the line.
    #[test]
    fn a_window_source_decline_names_the_whole_identity() {
        let decline = EngineFacadeDecline::WindowSourceDisappearedBeforePin {
            identity: identity(),
        };
        assert_eq!(
            decline.fields(),
            vec![
                ("identity_kind", "surface".into()),
                ("identity_id", "7".into()),
                ("identity_width", "64".into()),
                ("identity_height", "32".into()),
                ("identity_generation", "9".into()),
                ("identity_format", "Bgra8".into()),
            ]
        );
    }
}
