//! Host-memory topology changes placement and transfer scheduling only.
//!
//! The semantic core decides resource identity, content versions, completion,
//! and guest-visible success before this policy is consulted. Host-pointer
//! import is deliberately absent from this interface: it is an orthogonal
//! measured capability and each placement policy must work both with and
//! without it.

mod discrete;
mod unified;

use crate::memory::{MemoryClass, MemoryRequest, MemoryTopology};

/// Unified-memory default for the maximum draws retained in one batch.
pub const UNIFIED_DEFAULT_BATCH_DRAWS: u64 = 128;
/// Discrete-memory default for the maximum draws retained in one batch.
pub const DISCRETE_DEFAULT_BATCH_DRAWS: u64 = 32;
/// Largest topology-selected default accepted by the batching control.
pub const MAX_BATCH_DRAWS: u64 = UNIFIED_DEFAULT_BATCH_DRAWS;

/// Decisions a host-memory topology may make.
trait TopologyPolicy {
    fn request(&self, class: MemoryClass) -> MemoryRequest;
    fn default_batch_draws(&self) -> u64;
}

/// Selected topology policy for one physical Vulkan device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolicyKind {
    Unified(unified::UnifiedMemoryPolicy),
    Discrete(discrete::DiscreteMemoryPolicy),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Sealed placement and batching policy selected from structural topology.
pub struct MemoryPlacementPolicy(PolicyKind);

impl MemoryPlacementPolicy {
    /// Select the policy for a classified physical-device topology.
    pub const fn new(topology: MemoryTopology) -> Self {
        match topology {
            MemoryTopology::Unified => Self(PolicyKind::Unified(unified::UnifiedMemoryPolicy)),
            MemoryTopology::Discrete => Self(PolicyKind::Discrete(discrete::DiscreteMemoryPolicy)),
        }
    }

    /// Build the Vulkan memory-property request for an allocation purpose.
    pub fn request(self, class: MemoryClass) -> MemoryRequest {
        match self.0 {
            PolicyKind::Unified(policy) => policy.request(class),
            PolicyKind::Discrete(policy) => policy.request(class),
        }
    }

    /// Default draw batch size for this topology.
    pub fn default_batch_draws(self) -> u64 {
        match self.0 {
            PolicyKind::Unified(policy) => policy.default_batch_draws(),
            PolicyKind::Discrete(policy) => policy.default_batch_draws(),
        }
    }
}

fn topology_independent_request(class: MemoryClass) -> Option<MemoryRequest> {
    use ash::vk::MemoryPropertyFlags as F;
    match class {
        MemoryClass::DeviceLocal => Some(MemoryRequest {
            required: F::DEVICE_LOCAL,
            preferred: Vec::new(),
        }),
        MemoryClass::DeviceLocalPreferred => Some(MemoryRequest {
            required: F::empty(),
            preferred: vec![F::DEVICE_LOCAL],
        }),
        MemoryClass::Upload | MemoryClass::Readback => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::ContentState;
    use reims_vgpu_protocol::SubmissionId;
    use reims_vgpu_testkit::{assert_four_cell_guest_equivalence, GuestEffects};

    #[test]
    fn topology_cannot_change_device_local_semantics() {
        for class in [MemoryClass::DeviceLocal, MemoryClass::DeviceLocalPreferred] {
            assert_eq!(
                MemoryPlacementPolicy::new(MemoryTopology::Unified).request(class),
                MemoryPlacementPolicy::new(MemoryTopology::Discrete).request(class)
            );
        }
    }

    #[test]
    fn topology_policies_have_independent_submission_defaults() {
        assert_eq!(
            MemoryPlacementPolicy::new(MemoryTopology::Unified).default_batch_draws(),
            UNIFIED_DEFAULT_BATCH_DRAWS
        );
        assert_eq!(
            MemoryPlacementPolicy::new(MemoryTopology::Discrete).default_batch_draws(),
            DISCRETE_DEFAULT_BATCH_DRAWS
        );
    }

    #[test]
    fn all_four_memory_cells_preserve_one_semantic_trace() {
        let metrics = assert_four_cell_guest_equivalence(
            MemoryTopology::Unified,
            MemoryTopology::Discrete,
            |cell| {
                let policy = MemoryPlacementPolicy::new(cell.topology);
                let mut content = ContentState::default();
                let initial = content.current;
                if cell.host_pointer_import {
                    // An imported guest allocation is already a GPU-visible
                    // materialization of the stamped guest version.
                    content.gpu_materialized(initial).unwrap();
                } else {
                    // The staging rail reaches the same semantic state after
                    // its guest-to-GPU transfer completes.
                    content.copy_guest_to_gpu_completed(initial).unwrap();
                }
                let submission = SubmissionId::new(7);
                content.gpu_store_planned(submission).unwrap();
                let rendered = content.gpu_store_completed(submission).unwrap();
                content.copy_gpu_to_guest_completed(rendered).unwrap();
                let state = content.replicas;
                (
                    GuestEffects {
                        memory: vec![
                            content.current.get() as u8,
                            state.guest.map_or(0, |version| version.get() as u8),
                            state.gpu.map_or(0, |version| version.get() as u8),
                            state.host.map_or(0, |version| version.get() as u8),
                        ],
                        stamps: vec![(submission.get() as u32, rendered.get() as u32)],
                        interrupts: vec![submission.get() as u32],
                        refusals: Vec::new(),
                        presented: vec![0x30, 0x20, 0x10, 0xff],
                    },
                    (
                        policy.request(MemoryClass::Upload),
                        policy.request(MemoryClass::Readback),
                        policy.default_batch_draws(),
                        cell.host_pointer_import,
                    ),
                )
            },
        );

        assert_ne!(metrics[0].0, metrics[2].0, "upload placement may differ");
        assert_ne!(metrics[0].1, metrics[2].1, "readback placement may differ");
        assert!(metrics[0].3);
        assert!(!metrics[1].3);
        assert!(metrics[2].3);
        assert!(!metrics[3].3);
    }
}
