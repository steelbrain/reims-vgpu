//! Opaque ownership contracts for executor-local residents.

use reims_vgpu_protocol::{HeapObject, ResourceId, ResourceObject, StorageImageFormat};
use std::collections::BTreeMap;

/// Whether a retained guest-memory gather is licensed for identity reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatherVouch {
    /// The bytes cannot have changed since the retained gather was produced.
    Vouched,
    /// The bind must gather current bytes and publish a new identity.
    Fresh,
}

impl GatherVouch {
    pub const fn is_vouched(self) -> bool {
        matches!(self, Self::Vouched)
    }
}

/// Guest-semantic origin of a compute-resident texture.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ComputeStorageOrigin {
    Surface {
        mapping_id: u32,
        map_generation: u32,
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
    },
    Linear {
        resource: ResourceId<ResourceObject>,
        gva: u64,
        row_stride: u32,
        span_end: u64,
    },
    HeapPlacement {
        heap: ResourceId<HeapObject>,
        offset: u64,
        span_end: u64,
    },
    HeapAllocation {
        heap: ResourceId<HeapObject>,
        allocation: ResourceId<ResourceObject>,
    },
}

/// Exact protocol-backed compute storage-image view eligible for residency.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComputeStorageResidencyKey {
    pub origin: ComputeStorageOrigin,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
}

impl ComputeStorageResidencyKey {
    #[allow(
        clippy::too_many_arguments,
        reason = "the key constructor names every contract identity component"
    )]
    pub fn surface(
        mapping_id: u32,
        map_generation: u32,
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::Surface {
                mapping_id,
                map_generation,
                surface_offset,
                surface_bpr,
                span_end,
            },
            width,
            height,
            pixel_format,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the key constructor names every contract identity component"
    )]
    pub fn linear(
        resource: ResourceId<ResourceObject>,
        gva: u64,
        row_stride: u32,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::Linear {
                resource,
                gva,
                row_stride,
                span_end,
            },
            width,
            height,
            pixel_format,
        }
    }

    pub fn heap_placement(
        heap: ResourceId<HeapObject>,
        offset: u64,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::HeapPlacement {
                heap,
                offset,
                span_end,
            },
            width,
            height,
            pixel_format,
        }
    }

    pub fn heap_allocation(
        heap: ResourceId<HeapObject>,
        allocation: ResourceId<ResourceObject>,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            origin: ComputeStorageOrigin::HeapAllocation { heap, allocation },
            width,
            height,
            pixel_format,
        }
    }

    pub fn is_linear(&self) -> bool {
        matches!(self.origin, ComputeStorageOrigin::Linear { .. })
    }

    pub fn is_heap(&self) -> bool {
        matches!(
            self.origin,
            ComputeStorageOrigin::HeapPlacement { .. }
                | ComputeStorageOrigin::HeapAllocation { .. }
        )
    }

    pub fn surface_window(&self) -> Option<(u32, u64, u64)> {
        match self.origin {
            ComputeStorageOrigin::Surface {
                mapping_id,
                surface_offset,
                span_end,
                ..
            } => Some((mapping_id, surface_offset, span_end)),
            ComputeStorageOrigin::Linear { .. }
            | ComputeStorageOrigin::HeapPlacement { .. }
            | ComputeStorageOrigin::HeapAllocation { .. } => None,
        }
    }

    pub fn linear_window(&self) -> Option<(ResourceId<ResourceObject>, u64, u32, u64)> {
        match self.origin {
            ComputeStorageOrigin::Linear {
                resource,
                gva,
                row_stride,
                span_end,
            } => Some((resource, gva, row_stride, span_end)),
            ComputeStorageOrigin::Surface { .. }
            | ComputeStorageOrigin::HeapPlacement { .. }
            | ComputeStorageOrigin::HeapAllocation { .. } => None,
        }
    }

    pub fn resource(&self) -> Option<ResourceId<ResourceObject>> {
        match self.origin {
            ComputeStorageOrigin::Linear { resource, .. } => Some(resource),
            ComputeStorageOrigin::HeapAllocation { allocation, .. } => Some(allocation),
            ComputeStorageOrigin::Surface { .. } | ComputeStorageOrigin::HeapPlacement { .. } => {
                None
            }
        }
    }

    pub fn heap(&self) -> Option<ResourceId<HeapObject>> {
        match self.origin {
            ComputeStorageOrigin::HeapPlacement { heap, .. }
            | ComputeStorageOrigin::HeapAllocation { heap, .. } => Some(heap),
            ComputeStorageOrigin::Surface { .. } | ComputeStorageOrigin::Linear { .. } => None,
        }
    }
}

/// Semantic generations of compute-resident subresources.
///
/// This ledger states which executor generation still represents the current
/// guest-visible content. It does not own the native resident: the executor
/// answers that independently through [`ComputeResidencyService`]. Keeping the
/// two facts separate makes a lost native resident a typed execution outcome
/// without turning backend availability into content authority.
#[derive(Debug, Default)]
pub struct ComputeResidencyLedger {
    generations: BTreeMap<ComputeStorageResidencyKey, u32>,
}

impl ComputeResidencyLedger {
    pub fn generation(&self, key: &ComputeStorageResidencyKey) -> Option<u32> {
        if let Some(generation) = self.generations.get(key) {
            return Some(*generation);
        }
        let ComputeStorageOrigin::HeapPlacement { .. } = key.origin else {
            return None;
        };
        let mut aliases = self
            .generations
            .iter()
            .filter(|(candidate, _)| candidate.origin == key.origin)
            .map(|(_, generation)| *generation);
        let generation = aliases.next()?;
        aliases
            .all(|candidate| candidate == generation)
            .then_some(generation)
    }

    pub fn publish(&mut self, key: ComputeStorageResidencyKey, generation: u32) {
        if matches!(key.origin, ComputeStorageOrigin::HeapPlacement { .. }) {
            for (candidate, held_generation) in &mut self.generations {
                if candidate.origin == key.origin {
                    *held_generation = generation;
                }
            }
        }
        self.generations.insert(key, generation);
    }

    pub fn invalidate_surface_window(&mut self, mapping_id: u32, lo: u64, hi: u64) {
        self.generations.retain(|key, _| {
            key.surface_window().is_none_or(|(candidate, start, end)| {
                candidate != mapping_id || end <= lo || start >= hi
            })
        });
    }

    fn retire_where(
        &mut self,
        mut predicate: impl FnMut(&ComputeStorageResidencyKey) -> bool,
    ) -> Vec<ComputeStorageResidencyKey> {
        let retired: Vec<_> = self
            .generations
            .keys()
            .filter(|key| predicate(key))
            .copied()
            .collect();
        for key in &retired {
            self.generations.remove(key);
        }
        retired
    }

    /// Withdraw residents owned by one resource generation.
    pub fn retire_resource(
        &mut self,
        resource: ResourceId<ResourceObject>,
    ) -> Vec<ComputeStorageResidencyKey> {
        self.retire_where(|key| key.resource() == Some(resource))
    }

    /// Withdraw residents representing one exact storage origin.
    pub fn retire_origin(
        &mut self,
        origin: ComputeStorageOrigin,
    ) -> Vec<ComputeStorageResidencyKey> {
        self.retire_where(|key| key.origin == origin)
    }

    /// Withdraw every resident owned by one heap generation.
    pub fn retire_heap(&mut self, heap: ResourceId<HeapObject>) -> Vec<ComputeStorageResidencyKey> {
        self.retire_where(|key| key.heap() == Some(heap))
    }

    pub fn len(&self) -> usize {
        self.generations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.generations.is_empty()
    }

    pub fn contains(&self, key: &ComputeStorageResidencyKey) -> bool {
        self.generation(key).is_some()
    }
}

/// Persistent compute-image residency service.
pub trait ComputeResidencyService: std::fmt::Debug + Send + Sync {
    fn compute_resident_storage_generation(
        &self,
        _identity: &ComputeStorageResidencyKey,
    ) -> Option<u32> {
        None
    }

    fn compute_resident_sample_source(
        &self,
        _identity: &ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        None
    }

    fn unpin_resident_storage(&self, _identity: &ComputeStorageResidencyKey) {}

    fn retire_resident_storage_content(&self, _identity: &ComputeStorageResidencyKey) {}

    fn note_resident_storage_copied_out(&self, _identity: &ComputeStorageResidencyKey) {}
}

/// Backend-independent classification of a retained target's current content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentContentBacking {
    NotReady,
    /// The resident image is a view over the guest's canonical allocation.
    GuestAllocation,
    /// The resident image owns storage distinct from the guest allocation.
    DeviceAllocation,
}

#[cfg(test)]
mod tests {
    use super::{ComputeResidencyLedger, ComputeStorageOrigin, ComputeStorageResidencyKey};

    #[test]
    fn compute_residency_origins_are_disjoint_typed_identities() {
        let surface = ComputeStorageResidencyKey::surface(7, 2, 0, 64, 4096, 16, 16, 0x50);
        let resource = reims_vgpu_protocol::ResourceId::new(2, 7);
        let linear = ComputeStorageResidencyKey::linear(resource, 0, 64, 4096, 16, 16, 0x50);
        let heap_id = reims_vgpu_protocol::ResourceId::new(3, 5);
        let heap = ComputeStorageResidencyKey::heap_placement(heap_id, 0x100, 0x900, 16, 16, 0x50);

        assert_ne!(surface, linear);
        assert_ne!(linear, heap);
        assert!(matches!(
            surface.origin,
            ComputeStorageOrigin::Surface { .. }
        ));
        assert_eq!(surface.surface_window(), Some((7, 0, 4096)));
        assert_eq!(linear.resource(), Some(resource));
        assert!(heap.is_heap());
        assert_eq!(heap.heap(), Some(heap_id));
    }

    #[test]
    fn residency_invalidation_retires_only_intersecting_surface_windows() {
        let hit = ComputeStorageResidencyKey::surface(7, 1, 0, 16, 64, 4, 4, 0x50);
        let sibling = ComputeStorageResidencyKey::surface(7, 1, 128, 16, 192, 4, 4, 0x50);
        let heap = ComputeStorageResidencyKey::heap_placement(
            reims_vgpu_protocol::ResourceId::new(2, 1),
            0,
            64,
            4,
            4,
            0x50,
        );
        let mut ledger = ComputeResidencyLedger::default();
        ledger.publish(hit, 3);
        ledger.publish(sibling, 4);
        ledger.publish(heap, 5);

        ledger.invalidate_surface_window(7, 32, 96);

        assert_eq!(ledger.generation(&hit), None);
        assert_eq!(ledger.generation(&sibling), Some(4));
        assert_eq!(ledger.generation(&heap), Some(5));
    }

    #[test]
    fn reused_object_index_does_not_inherit_compute_residency() {
        let allocation = reims_vgpu_protocol::ResourceId::new(4, 1);
        let old = ComputeStorageResidencyKey::heap_allocation(
            reims_vgpu_protocol::ResourceId::new(9, 1),
            allocation,
            4,
            4,
            0x50,
        );
        let replacement = ComputeStorageResidencyKey::heap_allocation(
            reims_vgpu_protocol::ResourceId::new(9, 2),
            allocation,
            4,
            4,
            0x50,
        );
        let mut ledger = ComputeResidencyLedger::default();
        ledger.publish(old, 7);

        assert_eq!(ledger.generation(&old), Some(7));
        assert_eq!(ledger.generation(&replacement), None);
    }

    #[test]
    fn typed_lifetime_retirement_withdraws_only_its_residents() {
        let heap = reims_vgpu_protocol::ResourceId::new(9, 1);
        let other_heap = reims_vgpu_protocol::ResourceId::new(9, 2);
        let allocation = reims_vgpu_protocol::ResourceId::new(4, 1);
        let sibling_allocation = reims_vgpu_protocol::ResourceId::new(5, 1);
        let allocation_key =
            ComputeStorageResidencyKey::heap_allocation(heap, allocation, 4, 4, 0x50);
        let sibling_key =
            ComputeStorageResidencyKey::heap_allocation(heap, sibling_allocation, 4, 4, 0x50);
        let placement_key = ComputeStorageResidencyKey::heap_placement(heap, 0, 64, 4, 4, 0x50);
        let other_heap_key =
            ComputeStorageResidencyKey::heap_placement(other_heap, 0, 64, 4, 4, 0x50);
        let mut ledger = ComputeResidencyLedger::default();
        for key in [allocation_key, sibling_key, placement_key, other_heap_key] {
            ledger.publish(key, 3);
        }

        assert_eq!(ledger.retire_resource(allocation), vec![allocation_key]);
        assert_eq!(
            ledger.retire_origin(placement_key.origin),
            vec![placement_key]
        );
        assert_eq!(ledger.retire_heap(heap), vec![sibling_key]);
        assert_eq!(ledger.len(), 1);
        assert!(ledger.contains(&other_heap_key));
    }

    #[test]
    fn exact_heap_aliases_share_one_content_generation() {
        let heap = reims_vgpu_protocol::ResourceId::new(9, 1);
        let rgba = ComputeStorageResidencyKey::heap_placement(heap, 0, 16384, 64, 64, 0x46);
        let bgra = ComputeStorageResidencyKey::heap_placement(heap, 0, 16384, 64, 64, 0x50);
        let disjoint = ComputeStorageResidencyKey::heap_placement(heap, 16384, 32768, 64, 64, 0x46);
        let mut ledger = ComputeResidencyLedger::default();

        ledger.publish(rgba, 3);
        assert_eq!(
            ledger.generation(&bgra),
            Some(3),
            "a new view of the exact range inherits its current content"
        );
        assert_eq!(ledger.generation(&disjoint), None);

        ledger.publish(bgra, 4);
        assert_eq!(ledger.generation(&rgba), Some(4));
        assert_eq!(ledger.generation(&bgra), Some(4));
        assert_eq!(ledger.retire_origin(rgba.origin), vec![rgba, bgra]);
    }
}
