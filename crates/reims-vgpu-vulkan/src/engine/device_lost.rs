//! Typed Vulkan device-loss failures and recreate-policy exhaustion.
//!
//! A device loss is not one check: it can arrive from the draw queue, compute
//! queue, batched queue, or three fence operations, and the recreate policy
//! itself can exhaust or fail. These paths used to collapse into
//! `DrawError::DeviceLost(String)` and `vk_engine_device_lost_untyped`, leaving
//! the log unable to say which operation poisoned the device.

use ash::vk;

use reims_vgpu_observe::Decline;

use super::types::DrawError;
use super::vk_call::VkOp;

/// A Vulkan operation whose `ERROR_DEVICE_LOST` result triggers context
/// poisoning and bounded recreation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLostOp {
    DrawSubmit,
    ComputeSubmit,
    PoolsWaitFencesRetire,
    PoolsFenceStatusBeginEntry,
    PoolsWaitFencesEntry,
    PoolsSubmitBatch,
}

impl DeviceLostOp {
    /// The ordinary `VkCall` operation for a non-device-loss result at the same
    /// call site. The fence helper uses one enum for both outcomes so the two
    /// vocabularies cannot drift onto different rails.
    pub(crate) fn vk_op(self) -> VkOp {
        match self {
            Self::DrawSubmit => VkOp::ExecSubmit,
            Self::ComputeSubmit => VkOp::ComputeExecSubmit,
            Self::PoolsWaitFencesRetire => VkOp::PoolsWaitFencesRetire,
            Self::PoolsFenceStatusBeginEntry => VkOp::PoolsFenceStatusBeginEntry,
            Self::PoolsWaitFencesEntry => VkOp::PoolsWaitFencesEntry,
            Self::PoolsSubmitBatch => VkOp::PoolsSubmitBatch,
        }
    }
}

/// The specific cause that made the engine report device loss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceLostDecline {
    /// The bounded recreate budget was already exhausted.
    RecreateCapExhausted { cap: u32 },
    /// Creating the replacement context failed. The nested typed cause remains
    /// available instead of being flattened through `Display`.
    RecreateFailed { cause: Box<DrawError> },
    /// Test hook forced the draw rail to exercise device-loss recovery.
    ForcedDraw,
    /// Test hook forced the compute rail to exercise device-loss recovery.
    ForcedCompute,
    /// A concrete Vulkan operation returned `ERROR_DEVICE_LOST`.
    Driver {
        op: DeviceLostOp,
        result: vk::Result,
    },
}

impl DeviceLostDecline {
    fn squash(value: impl std::fmt::Display) -> String {
        value.to_string().replace(char::is_whitespace, "_")
    }
}

/// Set by any site that observes a lost device where it cannot run the recovery
/// itself, and consumed by the next site that can.
///
/// # Why a latch and not a direct call
///
/// The recovery needs the engine lock. The stamp-completion thread is the one
/// observer that must not take it — it exists to announce guest fences while the
/// drain worker holds the engine, and locking there would deadlock the pair it
/// was built to decouple. So it records the fact, and the drain's own
/// end-of-tranche flush acts on it; that runs about once a second whether or not
/// the guest is still submitting work.
///
/// That last clause is the whole point. Every other path in this backend was
/// written to "let the lost device surface on the next draw", which is correct
/// only while draws keep coming. On a driven macos-11 boot the loss surfaced
/// here, in `vkWaitSemaphores` — `stamp_wait_failed err=ERROR_DEVICE_LOST` —
/// and the guest then stopped drawing *because* of it: every leg after Maps
/// reported `draws=0` against a clean fail channel and a healthy 1 s census, and
/// `vk_device_recreate_proven` appears zero times in the whole boot. No draw was
/// ever going to come and surface it.
static DEVICE_LOST_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record that a lost device was seen somewhere that cannot recover from it.
pub(crate) fn note_device_lost_seen() {
    DEVICE_LOST_SEEN.store(true, std::sync::atomic::Ordering::Release);
}

/// Whether recovery work is waiting, without taking responsibility for it.
///
/// The end-of-tranche gate uses this only to decide whether to enter the engine
/// transaction. A loss arriving just after a `false` answer remains latched and
/// is consumed by the next tranche.
pub(crate) fn device_lost_seen() -> bool {
    DEVICE_LOST_SEEN.load(std::sync::atomic::Ordering::Acquire)
}

/// Take the latch if it is set. Callers must be able to run the recovery, since
/// taking it is what makes them responsible for it.
pub(crate) fn take_device_lost_seen() -> bool {
    DEVICE_LOST_SEEN.swap(false, std::sync::atomic::Ordering::AcqRel)
}

impl Decline for DeviceLostDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::RecreateCapExhausted { .. } => "vk_device_lost_recreate_cap_exhausted",
            Self::RecreateFailed { .. } => "vk_device_lost_recreate_failed",
            Self::ForcedDraw => "vk_device_lost_forced_draw",
            Self::ForcedCompute => "vk_device_lost_forced_compute",
            Self::Driver {
                op: DeviceLostOp::DrawSubmit,
                ..
            } => "vk_device_lost_exec_submit",
            Self::Driver {
                op: DeviceLostOp::ComputeSubmit,
                ..
            } => "vk_device_lost_compute_exec_submit",
            Self::Driver {
                op: DeviceLostOp::PoolsWaitFencesRetire,
                ..
            } => "vk_device_lost_pools_wait_fences_retire",
            Self::Driver {
                op: DeviceLostOp::PoolsFenceStatusBeginEntry,
                ..
            } => "vk_device_lost_pools_fence_status_begin_entry",
            Self::Driver {
                op: DeviceLostOp::PoolsWaitFencesEntry,
                ..
            } => "vk_device_lost_pools_wait_fences_entry",
            Self::Driver {
                op: DeviceLostOp::PoolsSubmitBatch,
                ..
            } => "vk_device_lost_pools_submit_batch",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::RecreateCapExhausted { cap } => vec![("cap", cap.to_string())],
            Self::RecreateFailed { cause } => {
                let mut fields = vec![("cause", cause.slug().to_string())];
                fields.extend(cause.fields());
                fields
            }
            Self::Driver { result, .. } => {
                vec![("vk_result", Self::squash(result))]
            }
            Self::ForcedDraw | Self::ForcedCompute => Vec::new(),
        }
    }
}

reims_vgpu_observe::decline_display!(DeviceLostDecline);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::init_decline::InitDecline;

    fn all() -> Vec<DeviceLostDecline> {
        vec![
            DeviceLostDecline::RecreateCapExhausted { cap: 3 },
            DeviceLostDecline::RecreateFailed {
                cause: Box::new(DrawError::Init(InitDecline::CreateDevice {
                    result: vk::Result::ERROR_INITIALIZATION_FAILED,
                })),
            },
            DeviceLostDecline::ForcedDraw,
            DeviceLostDecline::ForcedCompute,
            DeviceLostDecline::Driver {
                op: DeviceLostOp::DrawSubmit,
                result: vk::Result::ERROR_DEVICE_LOST,
            },
            DeviceLostDecline::Driver {
                op: DeviceLostOp::ComputeSubmit,
                result: vk::Result::ERROR_DEVICE_LOST,
            },
            DeviceLostDecline::Driver {
                op: DeviceLostOp::PoolsWaitFencesRetire,
                result: vk::Result::ERROR_DEVICE_LOST,
            },
            DeviceLostDecline::Driver {
                op: DeviceLostOp::PoolsFenceStatusBeginEntry,
                result: vk::Result::ERROR_DEVICE_LOST,
            },
            DeviceLostDecline::Driver {
                op: DeviceLostOp::PoolsWaitFencesEntry,
                result: vk::Result::ERROR_DEVICE_LOST,
            },
            DeviceLostDecline::Driver {
                op: DeviceLostOp::PoolsSubmitBatch,
                result: vk::Result::ERROR_DEVICE_LOST,
            },
        ]
    }

    #[test]
    fn every_device_loss_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_device_lost_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate device-loss slug");
    }

    #[test]
    fn driver_and_recreate_failures_keep_their_load_bearing_cause() {
        for decline in all() {
            let line = reims_vgpu_observe::Emit::decline("device_lost_test", &decline).render();
            assert!(line.starts_with(&format!("device_lost_test reason={}", decline.slug())));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }

        let recreate = &all()[1];
        let line = reims_vgpu_observe::Emit::decline("device_lost_test", recreate).render();
        assert!(
            line.contains("cause=vk_init_create_device vk_result="),
            "{line}"
        );
    }
}
