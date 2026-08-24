//! Arm mapper-service identity and capture lifecycle.
//!
//! Mapper references are 64-bit service identities. They do not name task
//! objects, registered surface backings, or GPU page-table mappings.

use std::collections::BTreeMap;

use reims_vgpu_protocol::{MapperRequestKind, MapperResolvedSurfaceId, MapperSurfaceRef};

/// Directed mapper capture published with one producer write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperCapture {
    /// Producer index that published this request (entry = producer - 1).
    pub producer: u32,
    pub mapper_device_kva: u64,
    pub request_kind: MapperRequestKind,
    /// Guest kernel address of the mapper-internal object.
    pub mapping_internal: u64,
}

/// Device-local state owned by the arm mapper service.
#[derive(Debug, Default)]
pub struct MapperService {
    pending_capture: Option<MapperCapture>,
    device_kva: u64,
    surfaces: BTreeMap<MapperSurfaceRef, MapperResolvedSurfaceId>,
}

impl MapperService {
    pub fn publish_capture(&mut self, capture: MapperCapture) {
        self.pending_capture = Some(capture);
    }

    /// Consume a capture only when it belongs to this published ring entry.
    pub fn take_capture(&mut self, producer: u32) -> Option<MapperCapture> {
        if self
            .pending_capture
            .is_some_and(|capture| capture.producer == producer)
        {
            self.pending_capture.take()
        } else {
            None
        }
    }

    pub fn restore_capture(&mut self, capture: MapperCapture) {
        self.pending_capture = Some(capture);
    }

    /// Zero cannot erase an already established mapper-device identity.
    pub fn observe_device(&mut self, device_kva: u64) {
        if device_kva != 0 {
            self.device_kva = device_kva;
        }
    }

    pub fn device_kva(&self) -> u64 {
        self.device_kva
    }

    pub fn map_surface(
        &mut self,
        mapper_surface: MapperSurfaceRef,
        surface: MapperResolvedSurfaceId,
    ) -> bool {
        if mapper_surface.get() == 0 {
            return false;
        }
        self.surfaces.insert(mapper_surface, surface);
        true
    }

    pub fn resolve_surface(
        &self,
        mapper_surface: MapperSurfaceRef,
    ) -> Option<MapperResolvedSurfaceId> {
        self.surfaces.get(&mapper_surface).copied()
    }

    pub fn retire_surface(&mut self, surface: MapperResolvedSurfaceId) {
        self.surfaces.retain(|_, related| *related != surface);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_consumption_is_scoped_to_the_publishing_entry() {
        let mut service = MapperService::default();
        let capture = MapperCapture {
            producer: 7,
            mapper_device_kva: 0x1000,
            request_kind: MapperRequestKind::Map,
            mapping_internal: 0x2000,
        };
        service.publish_capture(capture);
        assert_eq!(service.take_capture(6), None);
        assert_eq!(service.take_capture(7), Some(capture));
        assert_eq!(service.take_capture(7), None);
    }

    #[test]
    fn mapper_identity_is_wide_and_retirement_follows_the_resolved_surface() {
        let mut service = MapperService::default();
        let wide = MapperSurfaceRef::new(0x1_0000_0001);
        let low = MapperSurfaceRef::new(1);
        let surface = MapperResolvedSurfaceId::new(9);
        assert!(service.map_surface(wide, surface));
        assert_eq!(service.resolve_surface(wide), Some(surface));
        assert_eq!(service.resolve_surface(low), None);
        service.retire_surface(surface);
        assert_eq!(service.resolve_surface(wide), None);
    }
}
