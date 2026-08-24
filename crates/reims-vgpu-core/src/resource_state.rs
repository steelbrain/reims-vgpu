//! Resource-resolved state transitions carried outside the wire decoder.

use reims_vgpu_protocol::{ResourceId, ResourceObject, ResourceValidityOps, SurfaceId};

/// One decoded validity statement paired with the resource lifetime it names.
///
/// `resource` is absent when the statement names only resolved surface
/// mappings and no constructed task resource. Deferred pre-construction
/// currency is recorded before this resolved command is formed; no task-local
/// object reference crosses the execution boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedResourceState {
    pub resource: Option<ResourceId<ResourceObject>>,
    pub mappings: Box<[SurfaceId]>,
    pub ops: ResourceValidityOps,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_state_carries_only_generational_and_surface_identities() {
        let update = ResolvedResourceState {
            resource: None,
            mappings: vec![SurfaceId::new(7)].into_boxed_slice(),
            ops: ResourceValidityOps::PAGE_ON,
        };
        assert_eq!(update.resource, None);
        assert_eq!(update.mappings.as_ref(), [SurfaceId::new(7)]);
    }
}
