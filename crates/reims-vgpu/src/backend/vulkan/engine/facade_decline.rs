//! Typed failures at the Vulkan engine façade and host-window presenter seam.
//!
//! These checks are neither malformed draw/compute requests nor failed Vulkan
//! calls. They reject an engine entry point because the façade's tracked state
//! disappeared or disagreed with the caller — the named resident is absent, is
//! at the wrong generation, or is not yet content-ready.

use super::compute_execution::residency_fields;
use super::draw_execution::identity_fields;
use super::types::TargetIdentity;
use crate::model::ComputeStorageResidencyKey;
use crate::observe::Decline;

/// A specific engine façade or host-window presenter state failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineFacadeDecline {
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
}

impl Decline for EngineFacadeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WindowPresenterNotAttached => "vk_engine_window_presenter_not_attached",
            Self::StorageReadResidentAbsent { .. } => "vk_engine_storage_read_resident_absent",
            Self::StorageReadGenerationMismatch { .. } => {
                "vk_engine_storage_read_generation_mismatch"
            }
            Self::WindowSourceDisappearedBeforePin { .. } => {
                "vk_engine_window_source_disappeared_before_pin"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
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
        }
    }
}

crate::observe::decline_display!(EngineFacadeDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn residency() -> ComputeStorageResidencyKey {
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

    fn identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 7,
            width: 64,
            height: 32,
            generation: 9,
            format: crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
        }
    }

    fn all() -> Vec<EngineFacadeDecline> {
        vec![
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
        assert_eq!(before, 4, "the engine façade reason census moved");
        assert_eq!(before, slugs.len(), "duplicate engine façade slug");
    }

    #[test]
    fn engine_facade_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("engine_facade_test", &decline).render();
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
                (
                    "identity_format",
                    format!(
                        "{:?}",
                        crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT
                    )
                ),
            ]
        );
    }
}
