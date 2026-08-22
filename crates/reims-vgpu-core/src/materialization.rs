//! Guest-lifetime ownership for retained host materializations.
//!
//! Values are generic because a host adapter owns their concrete import and
//! alias types. Keys, address coverage, and retirement are semantic: they are
//! driven only by task/object lifetime and decoded mapping changes.

use std::collections::BTreeMap;

use reims_vgpu_protocol::{
    ByteLength, ByteOffset, GuestVirtualAddress, ObjectTableRef, ResourceObject, TaskId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterializationOwner {
    pub task: TaskId,
    pub object: ObjectTableRef<ResourceObject>,
}

impl MaterializationOwner {
    pub const fn new(task: TaskId, object: ObjectTableRef<ResourceObject>) -> Self {
        Self { task, object }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundWindowKey {
    pub owner: MaterializationOwner,
    pub offset: ByteOffset,
    pub extent_cap: Option<ByteLength>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestAddressSpan {
    pub start: GuestVirtualAddress,
    pub length: ByteLength,
}

impl GuestAddressSpan {
    pub const fn new(start: GuestVirtualAddress, length: ByteLength) -> Self {
        Self { start, length }
    }

    pub fn overlaps(self, other: Self) -> bool {
        let a_len = self.length.get();
        let b_len = other.length.get();
        if a_len == 0 || b_len == 0 {
            return false;
        }
        let a_start = self.start.get();
        let b_start = other.start.get();
        a_start < b_start.saturating_add(b_len) && b_start < a_start.saturating_add(a_len)
    }
}

#[derive(Debug)]
struct Retained<T> {
    span: GuestAddressSpan,
    value: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaterializationShape {
    pub entries: usize,
    pub owners: usize,
    pub multi_offset_owners: usize,
    pub max_offsets: u32,
}

/// Values removed by one decoded guest-lifetime transition.
pub struct MaterializationRetirement<Resource> {
    pub window_count: usize,
    pub resources: Vec<Resource>,
}

/// Unbounded retained materializations tied to decoded guest lifetimes.
#[derive(Debug)]
pub struct MaterializationRegistry<Window, Resource> {
    windows: BTreeMap<BoundWindowKey, Retained<Window>>,
    resources: BTreeMap<MaterializationOwner, Retained<Resource>>,
}

impl<Window, Resource> Default for MaterializationRegistry<Window, Resource> {
    fn default() -> Self {
        Self {
            windows: BTreeMap::new(),
            resources: BTreeMap::new(),
        }
    }
}

impl<Window, Resource> MaterializationRegistry<Window, Resource> {
    fn take_where(
        &mut self,
        mut window: impl FnMut(BoundWindowKey, GuestAddressSpan) -> bool,
        mut resource: impl FnMut(MaterializationOwner, GuestAddressSpan) -> bool,
    ) -> MaterializationRetirement<Resource> {
        let mut window_count = 0;
        let mut kept_windows = BTreeMap::new();
        for (key, entry) in std::mem::take(&mut self.windows) {
            if window(key, entry.span) {
                window_count += 1;
            } else {
                kept_windows.insert(key, entry);
            }
        }
        self.windows = kept_windows;

        let mut resources = Vec::new();
        let mut kept_resources = BTreeMap::new();
        for (owner, entry) in std::mem::take(&mut self.resources) {
            if resource(owner, entry.span) {
                resources.push(entry.value);
            } else {
                kept_resources.insert(owner, entry);
            }
        }
        self.resources = kept_resources;
        MaterializationRetirement {
            window_count,
            resources,
        }
    }

    pub fn take_task(&mut self, task: TaskId) -> MaterializationRetirement<Resource> {
        self.take_where(
            |key, _| key.owner.task == task,
            |owner, _| owner.task == task,
        )
    }

    pub fn take_object(
        &mut self,
        owner: MaterializationOwner,
    ) -> MaterializationRetirement<Resource> {
        self.take_where(
            |key, _| key.owner == owner,
            |candidate, _| candidate == owner,
        )
    }

    pub fn take_range(
        &mut self,
        task: TaskId,
        span: GuestAddressSpan,
    ) -> MaterializationRetirement<Resource> {
        self.take_where(
            |key, entry| key.owner.task == task && entry.overlaps(span),
            |owner, entry| owner.task == task && entry.overlaps(span),
        )
    }

    pub fn take_all(&mut self) -> MaterializationRetirement<Resource> {
        self.take_where(|_, _| true, |_, _| true)
    }

    pub fn window(&self, key: BoundWindowKey) -> Option<&Window> {
        self.windows.get(&key).map(|entry| &entry.value)
    }

    pub fn insert_window(&mut self, key: BoundWindowKey, span: GuestAddressSpan, value: Window) {
        self.windows.insert(key, Retained { span, value });
    }

    pub fn resource(&self, owner: MaterializationOwner) -> Option<&Resource> {
        self.resources.get(&owner).map(|entry| &entry.value)
    }

    pub fn resource_mut(&mut self, owner: MaterializationOwner) -> Option<&mut Resource> {
        self.resources.get_mut(&owner).map(|entry| &mut entry.value)
    }

    /// Borrow every resource-shaped materialization for read-only census work.
    ///
    /// The iterator exposes neither keys nor mutation: ownership and retirement
    /// remain operations of this registry, while an adapter can still measure
    /// properties of its concrete resource payloads without maintaining a
    /// second index that could drift from this one.
    pub fn resource_values(&self) -> impl Iterator<Item = &Resource> {
        self.resources.values().map(|entry| &entry.value)
    }

    pub fn insert_resource(
        &mut self,
        owner: MaterializationOwner,
        span: GuestAddressSpan,
        value: Resource,
    ) -> Option<Resource> {
        self.resources
            .insert(owner, Retained { span, value })
            .map(|entry| entry.value)
    }

    /// Retire one task's materializations. The count preserves the existing
    /// bind-window census; whole-resource retirement is still performed but is
    /// not misreported as a draw-path resolution.
    pub fn retire_task(&mut self, task: TaskId) -> usize {
        self.take_task(task).window_count
    }

    pub fn retire_object(&mut self, owner: MaterializationOwner) -> usize {
        self.take_object(owner).window_count
    }

    pub fn retire_range(&mut self, task: TaskId, span: GuestAddressSpan) -> usize {
        self.take_range(task, span).window_count
    }

    pub fn clear(&mut self) {
        let _ = self.take_all();
    }

    pub fn window_len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty() && self.resources.is_empty()
    }

    pub fn shape(&self) -> MaterializationShape {
        let mut per_owner = BTreeMap::<MaterializationOwner, u32>::new();
        for key in self.windows.keys() {
            *per_owner.entry(key.owner).or_default() += 1;
        }
        MaterializationShape {
            entries: self.windows.len(),
            owners: per_owner.len(),
            multi_offset_owners: per_owner.values().filter(|count| **count > 1).count(),
            max_offsets: per_owner.values().copied().max().unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(task: u32, object: u32) -> MaterializationOwner {
        MaterializationOwner::new(
            TaskId::new(task),
            ObjectTableRef::<ResourceObject>::new(object),
        )
    }

    fn span(start: u64, length: u64) -> GuestAddressSpan {
        GuestAddressSpan::new(GuestVirtualAddress::new(start), ByteLength::new(length))
    }

    fn key(owner: MaterializationOwner, offset: u64) -> BoundWindowKey {
        BoundWindowKey {
            owner,
            offset: ByteOffset::new(offset),
            extent_cap: None,
        }
    }

    #[test]
    fn object_retirement_does_not_touch_an_equal_reference_in_another_task() {
        let mut registry = MaterializationRegistry::<u8, u8>::default();
        registry.insert_window(key(owner(1, 7), 0), span(0x1000, 0x1000), 1);
        registry.insert_window(key(owner(2, 7), 0), span(0x1000, 0x1000), 2);
        registry.insert_resource(owner(1, 7), span(0x1000, 0x1000), 3);

        assert_eq!(registry.retire_object(owner(1, 7)), 1);
        assert_eq!(registry.window(key(owner(1, 7), 0)), None);
        assert_eq!(registry.resource(owner(1, 7)), None);
        assert_eq!(registry.window(key(owner(2, 7), 0)), Some(&2));
    }

    #[test]
    fn range_retirement_uses_semantic_address_coverage_for_both_payload_kinds() {
        let mut registry = MaterializationRegistry::<u8, u8>::default();
        let owner = owner(1, 3);
        registry.insert_window(key(owner, 0), span(0x1000, 0x1000), 1);
        registry.insert_window(key(owner, 1), span(0x4000, 0x1000), 2);
        registry.insert_resource(owner, span(0x1000, 0x2000), 3);

        assert_eq!(registry.retire_range(TaskId::new(1), span(0x1800, 1)), 1);
        assert_eq!(registry.window(key(owner, 0)), None);
        assert_eq!(registry.resource(owner), None);
        assert_eq!(registry.window(key(owner, 1)), Some(&2));
    }

    #[test]
    fn take_range_returns_the_owned_resource_value() {
        let mut registry = MaterializationRegistry::<u8, u32>::default();
        let owner = owner(1, 3);
        assert_eq!(
            registry.insert_resource(owner, span(0x1000, 0x2000), 0xfeed),
            None
        );

        let retired = registry.take_range(TaskId::new(1), span(0x1800, 1));
        assert_eq!(retired.window_count, 0);
        assert_eq!(retired.resources, vec![0xfeed]);
        assert!(registry.is_empty());
    }

    #[test]
    fn live_entries_have_no_capacity_retirement() {
        let mut registry = MaterializationRegistry::<u32, ()>::default();
        let owner = owner(4, 9);
        for offset in 0..2048 {
            registry.insert_window(key(owner, offset), span(offset, 1), offset as u32);
        }
        assert_eq!(registry.window_len(), 2048);
        assert_eq!(registry.shape().max_offsets, 2048);
    }

    #[test]
    fn resource_census_borrows_the_registrys_owned_values() {
        let mut registry = MaterializationRegistry::<(), u32>::default();
        registry.insert_resource(owner(1, 2), span(0x1000, 0x1000), 7);
        registry.insert_resource(owner(1, 3), span(0x2000, 0x1000), 11);

        assert_eq!(registry.resource_values().copied().sum::<u32>(), 18);
    }
}
