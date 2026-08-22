//! Guest task-directory lifetime and address-space roots.

use std::collections::BTreeMap;

/// One guest task directory and its optional object-list publication.
#[derive(Clone, Debug, Default)]
pub struct TaskEntry {
    pub active: bool,
    pub length: u64,
    pub directory_pfn: u32,
    pub object_list_pfn: u32,
    pub object_list_count: u32,
}

impl TaskEntry {
    /// A task the guest has defined but not yet given an object list.
    ///
    /// Object-list fields remain zero until the distinct object-list command
    /// publishes them; task definition does not invent a guest page or count.
    pub fn define(length: u64, directory_pfn: u32) -> Self {
        Self {
            active: true,
            length,
            directory_pfn,
            object_list_pfn: 0,
            object_list_count: 0,
        }
    }
}

/// Live tasks keyed by the guest's complete `u32` task namespace.
///
/// There is no host-selected capacity. Entries live and die with guest task
/// definition/deletion, and iteration preserves ascending task-id order.
#[derive(Clone, Debug, Default)]
pub struct TaskTable(BTreeMap<u32, TaskEntry>);

impl TaskTable {
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn get(&self, id: u32) -> Option<&TaskEntry> {
        self.0.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut TaskEntry> {
        self.0.get_mut(&id)
    }

    pub fn is_active(&self, id: u32) -> bool {
        self.get(id).is_some_and(|task| task.active)
    }

    pub fn define(&mut self, id: u32, entry: TaskEntry) {
        self.0.insert(id, entry);
    }

    pub fn remove(&mut self, id: u32) {
        self.0.remove(&id);
    }

    pub fn live(&self) -> impl Iterator<Item = (u32, &TaskEntry)> {
        self.0
            .iter()
            .filter(|(_, task)| task.active)
            .map(|(&id, task)| (id, task))
    }

    pub fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.live().map(|(id, _)| id)
    }

    pub fn live_count(&self) -> usize {
        self.live_ids().count()
    }
}

/// Fixture convenience for tests which mutate an already-defined task.
#[cfg(feature = "test-fixtures")]
impl std::ops::Index<u32> for TaskTable {
    type Output = TaskEntry;

    fn index(&self, id: u32) -> &TaskEntry {
        self.get(id)
            .unwrap_or_else(|| panic!("test indexed task {id}, which nothing defined"))
    }
}

#[cfg(feature = "test-fixtures")]
impl std::ops::IndexMut<u32> for TaskTable {
    fn index_mut(&mut self, id: u32) -> &mut TaskEntry {
        self.get_mut(id)
            .unwrap_or_else(|| panic!("test indexed task {id}, which nothing defined"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_width_task_ids_have_guest_owned_lifetimes() {
        let mut tasks = TaskTable::new();
        tasks.define(u32::MAX, TaskEntry::define(0x4000, 7));
        assert!(tasks.is_active(u32::MAX));
        assert_eq!(tasks.live_ids().collect::<Vec<_>>(), vec![u32::MAX]);

        tasks.remove(u32::MAX);
        assert!(!tasks.is_active(u32::MAX));
    }

    #[test]
    fn task_definition_does_not_invent_an_object_list() {
        let task = TaskEntry::define(0x8000, 9);
        assert_eq!((task.object_list_pfn, task.object_list_count), (0, 0));
    }
}
