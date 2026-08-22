use super::{
    topology_independent_request, MemoryClass, MemoryRequest, TopologyPolicy,
    DISCRETE_DEFAULT_BATCH_DRAWS,
};
use ash::vk::MemoryPropertyFlags as F;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DiscreteMemoryPolicy;

impl TopologyPolicy for DiscreteMemoryPolicy {
    fn request(&self, class: MemoryClass) -> MemoryRequest {
        if let Some(request) = topology_independent_request(class) {
            return request;
        }
        match class {
            MemoryClass::Upload => MemoryRequest {
                required: F::HOST_VISIBLE | F::HOST_COHERENT,
                preferred: Vec::new(),
            },
            MemoryClass::Readback => MemoryRequest {
                required: F::HOST_VISIBLE,
                preferred: vec![F::HOST_CACHED, F::HOST_COHERENT],
            },
            MemoryClass::DeviceLocal | MemoryClass::DeviceLocalPreferred => unreachable!(),
        }
    }

    fn default_batch_draws(&self) -> u64 {
        DISCRETE_DEFAULT_BATCH_DRAWS
    }
}
